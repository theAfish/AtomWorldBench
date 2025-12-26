import requests
import uuid
import json
from typing import List, Optional, Dict, Any
from .base_model import BaseModel

class GoogleADKModel(BaseModel):
    def __init__(
        self, 
        model_name: str, 
        agent_url: str = "http://localhost:8000", 
        app_name: str = "default_agent",
        user_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Google ADK Model wrapper.
        
        Args:
            model_name: The name of the model.
            agent_url: The base URL of the ADK API server (e.g., http://localhost:8000).
            app_name: The name of the agent application (folder name).
            user_id: Optional user ID. If not provided, a random one is generated.
            **kwargs: Additional arguments.
        """
        super().__init__(model_name, **kwargs)
        self.agent_url = agent_url.rstrip('/')
        self.app_name = app_name
        self.user_id = user_id if user_id else f"user_{uuid.uuid4().hex[:8]}"
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the Google ADK agent.
        
        Args:
            prompt: The input text prompt.
            **kwargs: Additional arguments (e.g., session_id).
            
        Returns:
            The generated text response.
        """
        # Use provided session_id or generate a new one for each request to ensure fresh state if needed
        session_id = kwargs.get("session_id", f"session_{uuid.uuid4().hex[:8]}")
        
        # 1. Create the session first (required by ADK)
        create_session_url = f"{self.agent_url}/apps/{self.app_name}/users/{self.user_id}/sessions/{session_id}"
        try:
            # Attempt to create the session
            session_resp = requests.post(create_session_url, json={}, headers={"Content-Type": "application/json"})
            
            if session_resp.status_code == 404:
                print(f"Error: Session creation returned 404 at {create_session_url}. Check if 'app_name' ({self.app_name}) matches your agent's folder name.")
            elif session_resp.status_code >= 400 and session_resp.status_code != 409:
                # 409 means Conflict (Session already exists), which is acceptable.
                print(f"Warning: Session creation failed with status {session_resp.status_code}: {session_resp.text}")
                
        except Exception as e:
            print(f"Warning: Failed to connect to create session at {create_session_url}: {e}")

        # 2. Run the agent
        url = f"{self.agent_url}/run"
        
        payload = {
            "appName": self.app_name,
            "userId": self.user_id,
            "sessionId": session_id,
            "newMessage": {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            events = response.json()
            
            # Parse the events to extract the model response
            model_response_text = ""
            
            if isinstance(events, list):
                for event in events:
                    content = event.get("content", {})
                    role = content.get("role")
                    
                    if role == "model":
                        parts = content.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                model_response_text += part["text"]
                                
            return model_response_text.strip()
            
        except Exception as e:
            print(f"Error calling Google ADK agent at {url}: {e}")
            return ""

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts.
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]