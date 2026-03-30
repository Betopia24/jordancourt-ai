import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import google.genai as genai
from google.genai import types
from langchain_core.runnables import Runnable
from langgraph.graph.message import add_messages
import re
from database import db
from s3_utils import s3_manager

load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if os.getenv("LOG_TO_FILE", "false").lower() == "true" else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Validate required environment variables on startup
REQUIRED_ENV_VARS = ["API_KEY", "MODEL_NAME", "POSTGRES_URL", "AWS_REGION", "AWS_S3_BUCKET_NAME"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# create a genai client instance
genai_client = genai.Client(api_key=os.getenv("API_KEY"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", 30))

# Input validation limits
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 5000))
MAX_AUDIO_SIZE = int(os.getenv("MAX_AUDIO_SIZE", 10 * 1024 * 1024))  # 10MB
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))  # 10MB
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mp3", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/bmp"}




class AudioProcessor(Runnable):
    async def invoke(self, input_data):
        # input_data: dict with 'audio_bytes', 'mime_type', 'prompt'
        parts = [
            types.Part.from_text(text=input_data['prompt']),
            types.Part.from_bytes(data=input_data['audio_bytes'], mime_type=input_data['mime_type'])
        ]
        # call the client generate_content API with correct async method
        resp = await genai_client.aio.models.generate_content(model=os.getenv("MODEL_NAME"), contents=parts)

        # try common response access patterns, return text when found
        if hasattr(resp, "text") and resp.text:
            return resp.text

        try:
            # handle dict-like or object outputs
            outputs = getattr(resp, "outputs", None) or getattr(resp, "response", None) or resp
            if isinstance(outputs, list):
                for out in outputs:
                    content = out.get("content") if isinstance(out, dict) else getattr(out, "content", None)
                    if content:
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                                return c.get("text")
                            # handle object content with .text
                            if hasattr(c, "text") and c.text:
                                return c.text

            # fallback: attempt to read candidates / first text-like field
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                if hasattr(cand, "content"):
                    return str(cand.content)
                if hasattr(cand, "text"):
                    return cand.text

        except Exception:
            pass

        # final fallback to stringified response
        return str(resp)


# Define the state for the conversation using MessagesState with custom reducer

class ChatSummary(BaseModel):
    user_unchangable_info: Annotated[str,Field(description="Information about the user that should not be changed.ex: user basic info like name,age,location etc., vehicle info like vehicle model,vehicle make,vehicle year etc., vehicle issues like engine problems,maintenance history etc.") ]
    summary: Annotated[str, Field(description="A brief summary of the conversation so far in two lines.")]
    key_points: Annotated[List[str], Field(description="A list of precise key points given by the user. The words should be in one word. The words should be precise an exact to the context of vehicle engines and diagnostics.")]
    what_worked: Annotated[List[str], Field(description="A list of things that worked well in the conversation. The words should be in one word.")]
    what_didnt_work: Annotated[List[str], Field(description="A list of things that did not work well in the conversation. The words should be in one word.")]

class ChatState(MessagesState):
    messages: Annotated[list, add_messages]
    image_context: str = None  # Store base64 image for persistence
    summary: ChatSummary = None  # Store conversation summary
    


# Initialize LLM
llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"), api_key=os.getenv("API_KEY"))

# System message for the chatbot
SYSTEM_MESSAGE = """You are an expert vehicle engine diagnostic assistant. Your role is to:
- Provide accurate, detailed information about vehicle engines, car maintenance, and automotive diagnostics
- Analyze engine problems and suggest solutions
- Explain technical concepts in a clear, understandable way
- Respond to basic conversational queries (greetings, personal questions) naturally and then guide the conversation back to vehicle topics
- Maintain a professional, helpful tone
- Provide responses without markdown formatting symbols (*, #, `, _, ~,"\n" etc.)
- Please avoid using any kind of newline or markdown formatting in your responses.
- The answer should be concise and to the point.
- Give the answer in minimum number of steps. and avoid unnecessary details.
- Give the answer as short as possible.
- The steps or the answers should be easy to understand for a layman.
- The answers should be strictly answered within minimum number of words and sentences.

You can engage in brief personal conversations, but your primary expertise is in vehicles and engines. After responding to personal queries, encourage users to ask vehicle-related questions.If the user continues to make irrelevant questions then remind him in a professioanl way he should stay in context."""

# Validation node to check if query is vehicle-related
async def validation_node(state: ChatState):
    message = state["messages"][-1]
    # Extract text from the message
    text_content = ""
    has_audio = False
    
    if isinstance(message.content, str):
        text_content = message.content
    elif isinstance(message.content, list):
        has_audio = any(item.get("type") == "audio" for item in message.content)
        for item in message.content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content = item.get("text", "")
                break
    
    # Skip validation for audio-only queries (audio content is harder to pre-validate)
    if has_audio and not text_content:
        return state
    
    # Check for basic conversational patterns that should always pass
    conversational_patterns = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "my name", "i am", "i'm", "who are you", "what is your name",
        "thank you", "thanks", "bye", "goodbye"
    ]
    if any(pattern in text_content.lower() for pattern in conversational_patterns):
        return state  # Allow basic conversational queries
    
    # Use LLM to validate if the query is vehicle-related with system context
    validation_messages = [
        SystemMessage(content="You are a query validator. Your job is to determine if queries are related to vehicles, engines, automotive maintenance, diagnostics, or basic conversation."),
        HumanMessage(content=f"""Determine if the following query should be allowed.

Query: {text_content}

Respond with only "YES" or "NO".
Rules:
- If the query is about vehicles, cars, engines, maintenance, diagnostics, or mechanics, respond "YES".
- If it's a greeting, personal introduction, or basic conversational query, respond "YES".
- If it references previous conversation topics, respond "YES".
- Only respond "NO" for queries completely unrelated to vehicles or basic conversation (e.g., math problems, cooking recipes, politics).""")
    ]
    
    try:
        validation_response = await llm.ainvoke(validation_messages)
        is_valid = "yes" in validation_response.content.lower()
        
        if not is_valid:
            return {"messages": [AIMessage(content="This is not a vehicle related question. Please give a valid question.")]}
    except Exception:
        pass  # If validation fails, proceed anyway
    
    return state

# Define the chatbot node
async def chatbot_node(state: ChatState):
    message = state["messages"][-1]
    
    # Check if this is an audio query
    is_audio_query = False
    if isinstance(message.content, list):
        is_audio_query = any(item.get("type") == "audio" for item in message.content)
    
    # Add persistent image context only for text/image queries (not audio)
    if not is_audio_query and state.get("image_context") and isinstance(message.content, list):
        has_image = any(item.get("type") == "image_url" for item in message.content)
        if not has_image:
            message.content.append({"type": "image_url", "image_url": {"url": state["image_context"]}})
    
    if is_audio_query:
        # Extract audio and use the AudioProcessor Runnable
        audio_item = next(item for item in message.content if item["type"] == "audio")
        text_item = next((item for item in message.content if item["type"] == "text"), None)
        prompt_text = (text_item.get("text") or "") if text_item else ""
        
        # Build prompt for audio processing with system context
        base_prompt = f"""{SYSTEM_MESSAGE}

Analyze the audio query about vehicle engines and provide a detailed, helpful response. Do not use markdown formatting."""
        if prompt_text:
            base_prompt = f"{base_prompt}\n\nAdditional context: {prompt_text}"
        
        input_data = {
            'prompt': base_prompt,
            'audio_bytes': audio_item["data"],
            'mime_type': audio_item["mime_type"]
        }
        response_text = await AudioProcessor().invoke(input_data)
        # Clean markdown symbols
        cleaned_response = re.sub(r'[*#`_~]', '', response_text)
        return {"messages": [AIMessage(content=cleaned_response)]}
    
    # Fallback to LangChain for text/image - inject system message
    try:
        # Prepend system message if not already present
        messages_with_system = state["messages"]
        if not any(isinstance(msg, SystemMessage) for msg in messages_with_system):
            messages_with_system = [SystemMessage(content=SYSTEM_MESSAGE)] + messages_with_system
        
        response = await llm.ainvoke(messages_with_system)
        # Clean markdown symbols from response
        cleaned_response = re.sub(r'[*#`_~]', '', response.content)
        return {"messages": [AIMessage(content=cleaned_response)]}
    except Exception as e:
        error_message = f"Sorry, the API is currently overloaded. Please try again later. Error: {str(e)}"
        return {"messages": [AIMessage(content=error_message)]}

# Summary node to summarize the last 20 messages
async def summary_node(state: ChatState):
    """
    Summarizes the last 20 messages in the conversation and stores the summary.
    """
    messages = state["messages"]
    
    # Get last 20 messages (excluding system messages)
    last_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)][-20:]
    
    if len(last_messages) < 2:
        # Not enough messages to summarize
        return state
    
    # Build conversation text for summarization
    conversation_text = ""
    for msg in last_messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = str(msg.content) if isinstance(msg.content, str) else "Multimodal content"
        conversation_text += f"{role}: {content}\n"
    
    # Create summarization prompt with structured output
    summary_prompt = f"""Analyze the following vehicle diagnostics conversation and provide a structured summary.

Conversation:
{conversation_text}

Provide:
1. A brief 2-line summary of the conversation like the stud
2. Key points (single words only, max 10 words)
3. Things that worked well in the diagnosis (single words only, max 5 words)
4. Things that didn't work or need attention (single words only, max 5 words)
5. Focus on vehicle-related issues, diagnostics, and solutions discussed.
example of structured summary:
context:
    name: Jordan
    user_details: California,USA,age-30
    vehicle_details: the vehicle is a 2020 Toyota Camry with 30,000 miles.
    problem[1]: engine overheating
    problem[2]: occasional stalling

"""
    
    try:
        # Use LLM with structured output
        llm_with_structure = llm.with_structured_output(ChatSummary)
        summary = await llm_with_structure.ainvoke([
            SystemMessage(content="You are a conversation summarizer specializing in vehicle diagnostics."),
            HumanMessage(content=summary_prompt)
        ])
        
        return {"summary": summary}
    except Exception as e:
        print(f"Summary generation error: {e}")
        # Return empty summary on error
        return state

# Build the graph
graph = StateGraph(ChatState)
graph.add_node("validation", validation_node)
graph.add_node("chatbot", chatbot_node)
graph.add_node("summary", summary_node)
graph.add_edge(START, "validation")
graph.add_edge("validation", "chatbot")
graph.add_edge("chatbot", END)

checkpointer = MemorySaver()
bot = graph.compile(checkpointer=checkpointer)

# FastAPI app
app = FastAPI(title="Vehicle Diagnostic AI Assistant", version="1.0.0")

# corrs
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     "https://dream-canvas.art",
    #     "https://xobehtedistuo-web-88238903740.us-central1.run.app",
    #     "http://localhost:3001"
    # ],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        await db.init_database()
        logger.info("Application started successfully")
        # Log configuration (without sensitive data)
        logger.info(f"Configuration: MAX_CONTEXT_MESSAGES={MAX_CONTEXT_MESSAGES}, MODEL={os.getenv('MODEL_NAME')}")
    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Close database pool on shutdown"""
    try:
        await db.close()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

@app.post("/chat")
async def chat(
    user_id: str = Form(...),
    chat_id: str = Form(...),
    message: str = Form(None),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    """
    Main chat endpoint supporting text, image, and audio queries.
    Each user can have multiple chats identified by chat_id.
    Images are stored in AWS S3.
    """
    try:
        # Input validation
        if not user_id or not chat_id:
            raise HTTPException(status_code=400, detail="user_id and chat_id are required")
        
        if len(user_id) > 255 or len(chat_id) > 255:
            raise HTTPException(status_code=400, detail="user_id and chat_id must be less than 255 characters")
        
        # Validate message length
        if message and len(message) > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters"
            )
        
        # Validate file uploads
        if image and (not image.filename or image.filename == '' or image.size == 0):
            image = None
        if audio and (not audio.filename or audio.filename == '' or audio.size == 0):
            audio = None
        
        # Validate image if provided
        if image:
            if image.size > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image size exceeds maximum of {MAX_IMAGE_SIZE / (1024*1024):.1f}MB"
                )
            
            content_type = image.content_type or ""
            if not any(allowed in content_type.lower() for allowed in ALLOWED_IMAGE_TYPES):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image type. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}"
                )
        
        # Validate audio if provided
        if audio:
            if audio.size > MAX_AUDIO_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio size exceeds maximum of {MAX_AUDIO_SIZE / (1024*1024):.1f}MB"
                )
            
            content_type = audio.content_type or ""
            if not any(allowed in content_type.lower() for allowed in ALLOWED_AUDIO_TYPES):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid audio type. Allowed: {', '.join(ALLOWED_AUDIO_TYPES)}"
                )
        
        if not message and not audio:
            raise HTTPException(status_code=400, detail="Either 'message' or 'audio' must be provided.")
        
        # Get conversation context from database (last 30 messages + summary)
        last_messages, summary_context = await db.get_last_n_messages(user_id, chat_id, MAX_CONTEXT_MESSAGES)
        
        # Build LangGraph config with unique thread
        config = {"configurable": {"thread_id": f"{user_id}_{chat_id}"}}
        
        # Retrieve existing state for image context
        existing_state = {}
        try:
            state_snapshot = bot.get_state(config)
            if state_snapshot and state_snapshot.values:
                existing_state = state_snapshot.values
        except Exception as e:
            logger.warning(f"Could not retrieve state: {e}")
        
        # Build current message content first
        current_content = []
        image_s3_url = None
        
        # Add text message
        if message:
            # Sanitize message (basic XSS prevention)
            sanitized_message = message.strip()
            if sanitized_message:
                current_content.append({"type": "text", "text": sanitized_message})
        
        # Handle image upload to S3 and use for processing
        if image:
            try:
                # Read image data
                image_data = await image.read()
                
                if not image_data:
                    raise HTTPException(status_code=400, detail="Image file is empty")
                
                # Get and increment image count for this chat
                image_count = await db.get_and_increment_image_count(user_id, chat_id)
                
                # Upload to S3 with original filename for format detection
                image_s3_url = s3_manager.upload_image(
                    image_data, 
                    user_id, 
                    chat_id, 
                    image_count,
                    filename=image.filename
                )
                
                if image_s3_url:
                    # Use base64 for LLM processing (temporary)
                    import base64
                    encoded = base64.b64encode(image_data).decode()
                    image_context = f"data:image/jpeg;base64,{encoded}"
                    current_content.append({"type": "image_url", "image_url": {"url": image_context}})
                    
                    # Store S3 URL in state for persistence
                    existing_state["image_context"] = image_context
                    existing_state["image_s3_url"] = image_s3_url
                else:
                    logger.error("Failed to upload image to S3")
                    raise HTTPException(status_code=500, detail="Failed to upload image. Please try again.")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Image processing error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to process image")
        else:
            # Use existing image context if available
            image_context = existing_state.get("image_context")
            image_s3_url = existing_state.get("image_s3_url")
            if image_context:
                current_content.append({"type": "image_url", "image_url": {"url": image_context}})
        
        # Handle audio
        if audio:
            try:
                audio_data = await audio.read()
                
                if not audio_data:
                    raise HTTPException(status_code=400, detail="Audio file is empty")
                
                mime_type = audio.content_type
                if not mime_type:
                    filename = audio.filename.lower() if audio.filename else ""
                    if filename.endswith('.wav'):
                        mime_type = "audio/wav"
                    elif filename.endswith('.mp3'):
                        mime_type = "audio/mp3"
                    elif filename.endswith('.aiff') or filename.endswith('.aif'):
                        mime_type = "audio/aiff"
                    elif filename.endswith('.aac'):
                        mime_type = "audio/aac"
                    elif filename.endswith('.ogg'):
                        mime_type = "audio/ogg"
                    elif filename.endswith('.flac'):
                        mime_type = "audio/flac"
                    else:
                        mime_type = "audio/wav"  # default
                
                # Transcribe audio to text
                input_data = {
                    'prompt': "Transcribe the audio to text. Provide only the transcription without any additional comments.",
                    'audio_bytes': audio_data,
                    'mime_type': mime_type
                }
                transcription = await AudioProcessor().invoke(input_data)
                transcription = re.sub(r'[*#`_~]', '', transcription).strip()
                
                if not transcription:
                    raise HTTPException(status_code=400, detail="Could not transcribe audio")
                
                if message:
                    message += " " + transcription
                else:
                    message = transcription
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Audio processing error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to process audio")
        
        # Save user message to database with image URL
        user_message_text = message if message else "[Audio query]"
        await db.save_message(user_id, chat_id, "user", user_message_text, image_s3_url)
        
        # Build messages list for LangGraph
        messages_list = []
        
        # Add summary context as system message if it exists
        if summary_context:
            messages_list.append(
                SystemMessage(content=f"Previous conversation summary: {summary_context}")
            )
        
        # Add historical messages from database
        for msg in last_messages:
            role = msg.get("role")
            content_text = msg.get("content")
            if role == "user":
                messages_list.append(HumanMessage(content=content_text))
            elif role == "assistant":
                messages_list.append(AIMessage(content=content_text))
        
        # Add the current message
        messages_list.append(HumanMessage(content=current_content))
        
        # Build state
        state = {
            "messages": messages_list,
            "image_context": existing_state.get("image_context"),
            "image_s3_url": image_s3_url
        }
        
        # Invoke the bot with full context
        result = await bot.ainvoke(state, config=config)
        bot_response = result["messages"][-1].content
        
        if not bot_response:
            bot_response = "I apologize, but I couldn't generate a response. Please try again."
        
        # Save bot response to database
        await db.save_message(user_id, chat_id, "assistant", bot_response)
        
        logger.info(f"Chat processed successfully: user_id={user_id}, chat_id={chat_id}, historical_msgs={len(last_messages)}, has_summary={summary_context is not None}, image_uploaded={image_s3_url is not None}")
        
        return {
            "user_id": user_id,
            "chat_id": chat_id,
            "response": bot_response,
            "image_url": image_s3_url,
            "context_used": {
                "historical_messages": len(last_messages),
                "has_summary": summary_context is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again later.")

@app.post("/chat/end/{user_id}/{chat_id}")
async def end_chat(user_id: str, chat_id: str):
    """
    End a chat session and create a summary for future context.
    This should be called when the user explicitly ends the conversation.
    Clears the chat state from memory after summarizing.
    """
    try:
        # Input validation
        if not user_id or not chat_id:
            raise HTTPException(status_code=400, detail="user_id and chat_id are required")
        
        if len(user_id) > 255 or len(chat_id) > 255:
            raise HTTPException(status_code=400, detail="Invalid user_id or chat_id length")
        
        config = {"configurable": {"thread_id": f"{user_id}_{chat_id}"}}
        
        # Get current state
        state_snapshot = bot.get_state(config)
        if not state_snapshot or not state_snapshot.values:
            raise HTTPException(status_code=404, detail="No active conversation found")
        
        # Generate summary
        summary_graph = StateGraph(ChatState)
        summary_graph.add_node("summary", summary_node)
        summary_graph.add_edge(START, "summary")
        summary_graph.add_edge("summary", END)
        summary_bot = summary_graph.compile(checkpointer=checkpointer)
        
        result = await summary_bot.ainvoke(state_snapshot.values, config=config)
        summary_obj = result.get("summary")
        
        if summary_obj:
            # Create summary text
            summary_text = f"{summary_obj.summary} Key points: {', '.join(summary_obj.key_points)}."
            await db.save_summary_context(user_id, chat_id, summary_text)
            
            # Clear the state from memory to free up RAM
            try:
                thread_id = f"{user_id}_{chat_id}"
                if hasattr(checkpointer, 'storage') and thread_id in checkpointer.storage:
                    del checkpointer.storage[thread_id]
                    logger.info(f"Chat state cleared from memory: thread_id={thread_id}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clear state from memory: {cleanup_error}")
            
            logger.info(f"Chat ended and summarized: user_id={user_id}, chat_id={chat_id}")
            
            return {
                "user_id": user_id,
                "chat_id": chat_id,
                "message": "Chat ended, summarized, and state cleared successfully",
            }
        
        return {"message": "Chat ended but summary generation failed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"End chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to end chat. Please try again.")

@app.get("/chats/{user_id}")
async def get_user_chats(user_id: str):
    """Get all chats for a specific user"""
    try:
        if not user_id or len(user_id) > 255:
            raise HTTPException(status_code=400, detail="Invalid user_id")
        
        chats = await db.get_user_chats(user_id)
        return {
            "user_id": user_id,
            "total_chats": len(chats),
            "chats": chats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get chats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chats")

@app.get("/chat/{user_id}/{chat_id}")
async def get_chat_history(user_id: str, chat_id: str):
    try:
        if not user_id or not chat_id or len(user_id) > 255 or len(chat_id) > 255:
            raise HTTPException(status_code=400, detail="Invalid user_id or chat_id")
        
        chat_data = await db.get_chat(user_id, chat_id)
        if not chat_data:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return {
            "user_id": user_id,
            "chat_id": chat_id,
            "messages": chat_data.get("messages", []),
            "summary_context": chat_data.get("summary_context"),
            "created_at": str(chat_data.get("created_at")),
            "updated_at": str(chat_data.get("updated_at"))
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get chat history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.delete("/chat/{user_id}/{chat_id}")
async def delete_chat(user_id: str, chat_id: str):
    """Delete a specific chat"""
    try:
        if not user_id or not chat_id or len(user_id) > 255 or len(chat_id) > 255:
            raise HTTPException(status_code=400, detail="Invalid user_id or chat_id")
        
        success = await db.delete_chat(user_id, chat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return {"message": "Chat deleted successfully", "user_id": user_id, "chat_id": chat_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete chat")

@app.get("/health")
async def health_check():
    """Health check endpoint with database connectivity check"""
    try:
        # Check database connectivity
        pool = await db.get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "service": "Vehicle Diagnostic AI Assistant",
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "Vehicle Diagnostic AI Assistant",
            "database": "disconnected",
            "error": str(e)
        }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True, log_level="info")