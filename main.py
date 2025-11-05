from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import os
from dotenv import load_dotenv
import httpx
import google.generativeai as genai
import json
from uuid import uuid4
import logging

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

app = FastAPI(
    title="Meeting Transcript Analyzer Agent",
    description="""
    An AI-powered agent that analyzes meeting transcripts and extracts actionable insights.
    
    ## Features
    
    * 📝 **Smart Summarization** - Generates concise meeting summaries
    * ✅ **Action Item Extraction** - Identifies tasks and assigns ownership
    * 🎯 **Decision Tracking** - Captures key decisions made during meetings
    * 👥 **Participant Recognition** - Identifies meeting participants
    * ➡️ **Next Steps** - Outlines follow-up actions
    * 🔗 **Telex.im Integration** - A2A protocol compliant
    
    ## A2A Protocol
    
    This agent implements the Agent-to-Agent (A2A) protocol based on JSON-RPC 2.0.
    Send requests to `/a2a/analyze` endpoint.
    
    ## API Documentation
    
    Use the interactive docs below to test the endpoints!
    """,
    version="1.0.0",
    contact={
        "name": "HNG Internship Stage 3",
        "url": "https://telex.im",
    },
    license_info={
        "name": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini with Flash model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Store conversation contexts
contexts: Dict[str, List[Dict]] = {}

# A2A Protocol Models
class MessagePart(BaseModel):
    kind: Literal["text", "data", "file"]
    text: Optional[str] = None
    data: Optional[Dict[str, Any] | List[Any]] = None  # Allow both dict and list
    file_url: Optional[str] = None

class A2AMessage(BaseModel):
    kind: Literal["message"] = "message"
    role: Literal["user", "agent", "system"]
    parts: List[MessagePart]
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    taskId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PushNotificationConfig(BaseModel):
    url: str
    token: Optional[str] = None
    authentication: Optional[Dict[str, Any]] = None

class MessageConfiguration(BaseModel):
    blocking: bool = True
    acceptedOutputModes: List[str] = Field(default=["text/plain", "application/json"])
    pushNotificationConfig: Optional[PushNotificationConfig] = None

class MessageParams(BaseModel):
    message: A2AMessage
    configuration: MessageConfiguration = Field(default_factory=MessageConfiguration)

class ExecuteParams(BaseModel):
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    messages: List[A2AMessage]

class A2AParams(BaseModel):
    """Union type for params - can be either MessageParams or ExecuteParams"""
    pass

class JSONRPCRequest(BaseModel):
    jsonrpc: Literal["2.0"]
    id: str
    method: Literal["message/send", "execute"]
    params: Dict[str, Any]  # Accept any dict, we'll parse it manually
    
    @property
    def message_params(self) -> Optional[MessageParams]:
        """Get params as MessageParams if method is message/send"""
        if self.method == "message/send":
            try:
                return MessageParams(**self.params)
            except:
                return None
        return None
    
    @property
    def execute_params(self) -> Optional[ExecuteParams]:
        """Get params as ExecuteParams if method is execute"""
        if self.method == "execute":
            try:
                return ExecuteParams(**self.params)
            except:
                return None
        return None

class TaskStatus(BaseModel):
    state: Literal["working", "completed", "input-required", "failed"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    message: Optional[A2AMessage] = None

class Artifact(BaseModel):
    artifactId: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    parts: List[MessagePart]

class TaskResult(BaseModel):
    id: str
    contextId: str
    status: TaskStatus
    artifacts: List[Artifact] = []
    history: List[A2AMessage] = []
    kind: Literal["task"] = "task"

class JSONRPCResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Optional[TaskResult] = None
    error: Optional[Dict[str, Any]] = None

# Legacy models for backward compatibility
class TelexMessage(BaseModel):
    text: str = Field(
        ...,
        description="The meeting transcript text to analyze",
        min_length=50,
        example="Team standup meeting - Nov 5, 2025\n\nJohn: Good morning everyone."
    )
    sender: Optional[str] = None
    channel_id: Optional[str] = None
    message_id: Optional[str] = None

class AnalysisResult(BaseModel):
    summary: str = Field(
        ...,
        description="A concise 2-3 sentence summary of the meeting",
        example="Team discussed project updates including completed authentication module and ongoing payment integration work. Key decision was made to use blue-green deployment strategy."
    )
    action_items: List[str] = Field(
        ...,
        description="List of specific tasks with owners if mentioned",
        example=["Sarah to review Mike's payment integration code on Friday", "Mike to complete payment integration by Thursday"]
    )
    key_decisions: List[str] = Field(
        ...,
        description="List of important decisions made during the meeting",
        example=["Use blue-green deployment for database migration"]
    )
    participants_mentioned: List[str] = Field(
        ...,
        description="List of participant names identified in the transcript",
        example=["John", "Sarah", "Mike"]
    )
    next_steps: List[str] = Field(
        ...,
        description="List of follow-up actions or next steps",
        example=["Next meeting scheduled for Monday at 10am"]
    )

class AnalysisResponse(BaseModel):
    success: bool = Field(..., description="Whether the analysis was successful")
    analysis: AnalysisResult = Field(..., description="The structured analysis results")
    original_length: int = Field(..., description="Length of the original transcript in characters")

class WebhookResponse(BaseModel):
    success: bool = Field(..., description="Whether the webhook processing was successful")
    message: str = Field(..., description="Status message")

class AgentInfo(BaseModel):
    agent: str = Field(..., description="Agent name")
    status: str = Field(..., description="Agent status")
    description: str = Field(..., description="Agent description")
    model: str = Field(..., description="AI model being used")
    version: str = Field(..., description="API version")

class HealthCheck(BaseModel):
    status: str = Field(..., description="Health status")
    model: str = Field(..., description="AI model name")

@app.get(
    "/",
    response_model=AgentInfo,
    summary="Get Agent Information",
    description="Returns basic information about the Meeting Transcript Analyzer agent",
    tags=["Info"]
)
async def root():
    """
    Get agent information including name, status, and capabilities.
    
    This endpoint provides metadata about the agent and can be used to verify
    that the service is running correctly.
    """
    logger.debug("📋 Agent info requested")
    return {
        "agent": "Meeting Transcript Analyzer",
        "status": "active",
        "description": "Analyzes meeting transcripts and extracts summaries, action items, and key decisions",
        "model": "gemini-2.0-flash-exp",
        "version": "1.0.0"
    }

@app.get(
    "/health",
    response_model=HealthCheck,
    summary="Health Check",
    description="Check if the agent is healthy and ready to process requests",
    tags=["Info"]
)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns the current health status of the agent and the AI model being used.
    """
    logger.debug("💚 Health check requested")
    return {
        "status": "healthy",
        "model": "gemini-2.0-flash-exp"
    }

class A2ARequestBody(BaseModel):
    """A2A Protocol Request Body for Swagger documentation"""
    jsonrpc: Literal["2.0"] = Field(..., example="2.0")
    id: str = Field(..., example="req-123")
    method: Literal["message/send", "execute"] = Field(..., example="message/send")
    params: Dict[str, Any] = Field(
        ...,
        example={
            "message": {
                "kind": "message",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Team standup - John completed auth module. Sarah will test tomorrow. We decided to launch next week."
                    }
                ],
                "messageId": "msg-456"
            },
            "configuration": {
                "blocking": True,
                "acceptedOutputModes": ["text/plain", "application/json"]
            }
        }
    )

@app.post(
    "/a2a/analyze",
    summary="A2A Protocol Endpoint",
    description="Main A2A endpoint for meeting transcript analysis following JSON-RPC 2.0",
    tags=["A2A Protocol"]
)
async def a2a_endpoint(body: A2ARequestBody):
    """
    A2A Protocol endpoint following JSON-RPC 2.0 specification.
    
    This endpoint implements the Agent-to-Agent protocol for seamless
    integration with Telex.im and other A2A-compliant platforms.
    
    **Supported Methods:**
    - `message/send`: Send a single message for analysis
    - `execute`: Execute with full context and message history
    
    **Example Request Body:**
    ```json
    {
      "jsonrpc": "2.0",
      "id": "req-123",
      "method": "message/send",
      "params": {
        "message": {
          "kind": "message",
          "role": "user",
          "parts": [
            {
              "kind": "text",
              "text": "Team standup - John completed auth module. Sarah will test tomorrow. We decided to launch next week."
            }
          ]
        },
        "configuration": {
          "blocking": true
        }
      }
    }
    ```
    
    **Response Format:**
    Returns a TaskResult with analysis artifacts and conversation history.
    """
    # Log raw request body
    logger.info("="*80)
    logger.info("📥 INCOMING A2A REQUEST - RAW BODY")
    logger.info("="*80)
    logger.info(f"Raw request body: {json.dumps(body.model_dump(), indent=2)}")
    logger.info("="*80)
    
    # Validate and parse A2A request
    try:
        rpc_request = JSONRPCRequest(
            jsonrpc=body.jsonrpc,
            id=body.id,
            method=body.method,
            params=body.params
        )
        logger.info("✅ Request validated successfully")
        logger.info(f"Request ID: {rpc_request.id}")
        logger.info(f"Method: {rpc_request.method}")
        logger.info(f"JSON-RPC Version: {rpc_request.jsonrpc}")
    except Exception as e:
        logger.error("="*80)
        logger.error("❌ REQUEST VALIDATION FAILED")
        logger.error("="*80)
        logger.error(f"Validation error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("="*80)
        return JSONResponse(
            status_code=422,
            content={
                "jsonrpc": "2.0",
                "id": body.id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request",
                    "data": {
                        "details": str(e),
                        "received_body": body.model_dump()
                    }
                }
            }
        )
    
    try:
        # Extract messages and configuration
        messages = []
        context_id = None
        task_id = None
        config = None
        
        if rpc_request.method == "message/send":
            # Parse message params
            msg_params = MessageParams(**rpc_request.params)
            messages = [msg_params.message]
            config = msg_params.configuration
            logger.info(f"📨 Method: message/send")
            logger.info(f"Message ID: {msg_params.message.messageId}")
            logger.info(f"Message Role: {msg_params.message.role}")
            logger.info(f"Message Parts: {len(msg_params.message.parts)}")
            for i, part in enumerate(msg_params.message.parts):
                logger.info(f"  Part {i+1}: kind={part.kind}")
                if part.kind == "text" and part.text:
                    text_preview = part.text[:100] + "..." if len(part.text) > 100 else part.text
                    logger.info(f"  Text preview: {text_preview}")
                elif part.kind == "data":
                    logger.info(f"  Data type: {type(part.data).__name__}")
            logger.info(f"Configuration: blocking={config.blocking}")
        elif rpc_request.method == "execute":
            # Parse execute params
            exec_params = ExecuteParams(**rpc_request.params)
            messages = exec_params.messages
            context_id = exec_params.contextId
            task_id = exec_params.taskId
            logger.info(f"📨 Method: execute")
            logger.info(f"Context ID: {context_id}")
            logger.info(f"Task ID: {task_id}")
            logger.info(f"Messages count: {len(messages)}")
        
        # Process messages
        logger.info("🔄 Processing messages...")
        result = await process_a2a_messages(
            messages=messages,
            context_id=context_id,
            task_id=task_id,
            config=config
        )
        logger.info(f"✅ Processing complete - Task ID: {result.id}, State: {result.status.state}")
        
        # Handle webhook notification if configured
        if config and not config.blocking and config.pushNotificationConfig:
            logger.info(f"📤 Sending webhook notification to: {config.pushNotificationConfig.url}")
            # Send async notification
            await send_webhook_notification(
                config.pushNotificationConfig.url,
                result,
                config.pushNotificationConfig.authentication
            )
            # Return immediate response
            logger.info("📨 Returning immediate 'working' response for async processing")
            return JSONRPCResponse(
                id=rpc_request.id,
                result=TaskResult(
                    id=result.id,
                    contextId=result.contextId,
                    status=TaskStatus(state="working"),
                    kind="task"
                )
            ).model_dump()
        
        # Build response
        response = JSONRPCResponse(
            id=rpc_request.id,
            result=result
        )
        
        logger.info("="*80)
        logger.info("📤 SENDING A2A RESPONSE")
        logger.info("="*80)
        logger.info(f"Response ID: {response.id}")
        logger.info(f"Task ID: {result.id}")
        logger.info(f"Context ID: {result.contextId}")
        logger.info(f"Status: {result.status.state}")
        logger.info(f"Artifacts count: {len(result.artifacts)}")
        logger.info(f"History count: {len(result.history)}")
        logger.info("="*80)
        
        return response.model_dump()
        
    except ValueError as e:
        logger.error("="*80)
        logger.error("❌ VALIDATION ERROR")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}")
        logger.error("="*80)
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": rpc_request.id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": {"details": str(e)}
                }
            }
        )
    except Exception as e:
        logger.error("="*80)
        logger.error("❌ INTERNAL ERROR")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("="*80)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": rpc_request.id,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": {"details": str(e)}
                }
            }
        )

async def process_a2a_messages(
    messages: List[A2AMessage],
    context_id: Optional[str] = None,
    task_id: Optional[str] = None,
    config: Optional[MessageConfiguration] = None
) -> TaskResult:
    """Process A2A messages and return TaskResult"""
    
    # Generate IDs if not provided
    context_id = context_id or str(uuid4())
    task_id = task_id or str(uuid4())
    
    logger.info(f"🔧 Processing - Context: {context_id}, Task: {task_id}")
    
    # Get or create context history
    if context_id not in contexts:
        contexts[context_id] = []
        logger.info(f"📝 Created new context: {context_id}")
    else:
        logger.info(f"📝 Using existing context: {context_id} (history: {len(contexts[context_id])} messages)")
    
    # Extract transcript from last user message
    user_message = messages[-1] if messages else None
    if not user_message:
        logger.error("❌ No message provided in request")
        raise ValueError("No message provided")
    
    # Extract transcript from text parts (handle multiple text parts)
    transcript_parts = []
    for part in user_message.parts:
        if part.kind == "text" and part.text:
            transcript_parts.append(part.text)
    
    transcript = " ".join(transcript_parts) if transcript_parts else ""
    
    logger.info(f"📄 Extracted {len(transcript_parts)} text parts")
    logger.info(f"📄 Total transcript length: {len(transcript)} characters")
    
    if not transcript or len(transcript.strip()) < 50:
        logger.warning(f"⚠️ Transcript too short: {len(transcript)} characters")
        # Return input-required status
        response_message = A2AMessage(
            role="agent",
            parts=[MessagePart(
                kind="text",
                text="👋 Hi! I'm the Meeting Transcript Analyzer.\n\nPlease send me a meeting transcript (at least 50 characters) and I'll provide:\n• Summary\n• Action items\n• Key decisions\n• Next steps"
            )],
            taskId=task_id
        )
        
        return TaskResult(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(
                state="input-required",
                message=response_message
            ),
            history=messages + [response_message]
        )
    
    # Process transcript with Gemini
    logger.info("🤖 Sending to Gemini for analysis...")
    analysis = await process_transcript(transcript)
    logger.info(f"✅ Gemini analysis complete:")
    logger.info(f"   - Action items: {len(analysis.action_items)}")
    logger.info(f"   - Key decisions: {len(analysis.key_decisions)}")
    logger.info(f"   - Participants: {len(analysis.participants_mentioned)}")
    logger.info(f"   - Next steps: {len(analysis.next_steps)}")
    
    # Create concise response message
    response_text = f"✅ Analysis complete. Found {len(analysis.action_items)} action items, {len(analysis.key_decisions)} key decisions, and {len(analysis.participants_mentioned)} participants."
    
    response_message = A2AMessage(
        role="agent",
        parts=[MessagePart(kind="text", text=response_text)],
        taskId=task_id
    )
    
    # Create artifacts with all the detailed information
    artifacts = [
        Artifact(
            name="summary",
            parts=[MessagePart(kind="text", text=analysis.summary)]
        ),
        Artifact(
            name="action_items",
            parts=[MessagePart(kind="data", data={"items": analysis.action_items})]
        ),
        Artifact(
            name="key_decisions",
            parts=[MessagePart(kind="data", data={"decisions": analysis.key_decisions})]
        ),
        Artifact(
            name="participants",
            parts=[MessagePart(kind="data", data={"participants": analysis.participants_mentioned})]
        ),
        Artifact(
            name="next_steps",
            parts=[MessagePart(kind="data", data={"steps": analysis.next_steps})]
        ),
        Artifact(
            name="analysis",
            parts=[MessagePart(kind="data", data=analysis.model_dump())]
        )
    ]
    
    # Only keep last user message and agent response in history (not full conversation)
    history = [messages[-1], response_message]
    contexts[context_id] = history
    
    return TaskResult(
        id=task_id,
        contextId=context_id,
        status=TaskStatus(
            state="completed",
            message=response_message
        ),
        artifacts=artifacts,
        history=history
    )

async def process_transcript(transcript: str) -> AnalysisResult:
    """Process transcript using Gemini to extract insights"""
    
    logger.info(f"🔍 Analyzing transcript ({len(transcript)} chars)...")
    
    prompt = f"""You are a meeting analysis assistant. Analyze the following meeting transcript and extract structured information.

Meeting Transcript:
{transcript}

Extract and return ONLY a valid JSON object with these exact keys:
- summary: A concise summary (2-3 sentences) as a STRING
- action_items: Array of STRINGS describing tasks (e.g., "James to optimize load time by Friday", "Mark to finish content migration by next week")
- key_decisions: Array of STRINGS describing decisions made
- participants_mentioned: Array of STRINGS with participant names only
- next_steps: Array of STRINGS describing next steps

IMPORTANT: 
- action_items must be an array of STRINGS, not objects
- Each action item should be a single string like "Person to do task by deadline"
- Do NOT use objects with task/owner fields

Return ONLY the JSON object, no other text.

Example format:
{{
  "summary": "Team discussed project status...",
  "action_items": ["James to optimize load time by Friday", "Mark to finish migration by next week"],
  "key_decisions": ["Decided to launch next Wednesday"],
  "participants_mentioned": ["Sarah", "James", "Mark"],
  "next_steps": ["Client demo next Wednesday"]
}}"""

    logger.info("📡 Calling Gemini API...")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=1024,
        )
    )
    logger.info("✅ Gemini API response received")
    
    # Extract JSON from response
    response_text = response.text.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    response_text = response_text.strip()
    
    try:
        result = json.loads(response_text)
        logger.info("✅ JSON parsed successfully")
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON parse failed, trying fallback extraction: {str(e)}")
        # Fallback: try to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            logger.info("✅ JSON extracted via fallback method")
        else:
            logger.error("❌ Could not parse JSON from Gemini response")
            logger.error(f"Response text: {response_text[:500]}")
            raise ValueError("Could not parse JSON from Gemini response")
    
    # Convert action items to strings if they're objects
    action_items = result.get("action_items", [])
    if action_items and isinstance(action_items[0], dict):
        logger.info("⚠️ Converting action items from objects to strings")
        action_items = [
            f"{item.get('owner', 'Someone')} to {item.get('task', 'complete task')}"
            if 'owner' in item and 'task' in item
            else str(item)
            for item in action_items
        ]
    
    return AnalysisResult(
        summary=result.get("summary", "No summary available"),
        action_items=action_items,
        key_decisions=result.get("key_decisions", []),
        participants_mentioned=result.get("participants_mentioned", []),
        next_steps=result.get("next_steps", [])
    )

def format_analysis_response(analysis: AnalysisResult) -> str:
    """Format analysis result for Telex message"""
    
    response = "📊 **Meeting Analysis Complete**\n\n"
    
    response += f"**📝 Summary:**\n{analysis.summary}\n\n"
    
    if analysis.action_items:
        response += "**✅ Action Items:**\n"
        for item in analysis.action_items:
            response += f"• {item}\n"
        response += "\n"
    
    if analysis.key_decisions:
        response += "**🎯 Key Decisions:**\n"
        for decision in analysis.key_decisions:
            response += f"• {decision}\n"
        response += "\n"
    
    if analysis.participants_mentioned:
        response += "**👥 Participants:**\n"
        response += ", ".join(analysis.participants_mentioned) + "\n\n"
    
    if analysis.next_steps:
        response += "**➡️ Next Steps:**\n"
        for step in analysis.next_steps:
            response += f"• {step}\n"
    
    return response

async def send_webhook_notification(
    webhook_url: str,
    result: TaskResult,
    auth: Optional[Dict[str, Any]] = None
):
    """Send result to webhook URL for async processing"""
    logger.info(f"📤 Sending webhook notification to: {webhook_url}")
    
    headers = {"Content-Type": "application/json"}
    
    if auth:
        if auth.get("schemes") == ["TelexApiKey"]:
            headers["Authorization"] = f"Bearer {auth.get('credentials')}"
            logger.info("🔐 Using TelexApiKey authentication")
        elif auth.get("token"):
            headers["Authorization"] = f"Bearer {auth.get('token')}"
            logger.info("🔐 Using token authentication")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                webhook_url,
                json=result.model_dump(),
                headers=headers,
                timeout=30.0
            )
            logger.info(f"✅ Webhook notification sent successfully: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Webhook notification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENT_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
