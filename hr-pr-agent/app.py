from huggingface_hub.inference._mcp.agent import Agent
from typing import Optional, Literal

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "microsoft/DialoGPT-medium")
DEFAULT_PROVIDER: Literal["hf-inference"] = "hf-inference"

# Global agent instance
agent_instance: Optional[Agent] = None

async def get_agent():
    """Get or create Agent instance"""
    print("🤖 get_agent() called...")
    global agent_instance
    if agent_instance is None and HF_TOKEN:
        print("🔧 Creating new Agent instance...")
        print(f"🔑 HF_TOKEN present: {bool(HF_TOKEN)}")
        print(f"🤖 Model: {HF_MODEL}")
        print(f"🔗 Provider: {DEFAULT_PROVIDER}")

        try:
            agent_instance = Agent(
                model=HF_MODEL,
                provider=DEFAULT_PROVIDER,
                api_key=HF_TOKEN,
                servers=[
                    {
                        "type": "stdio",
                        "config": {
                            "command": "python",
                            "args": ["mcp_server.py"],
                            "cwd": ".",
                            "env": {"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},
                        },
                    }
                ],
            )
            print("✅ Agent instance created successfully")
            print("🔧 Loading tools...")
            await agent_instance.load_tools()
            print("✅ Tools loaded successfully")
        except Exception as e:
            print(f"❌ Error creating/loading agent: {str(e)}")
            agent_instance = None



# Example of how the agent would use tools
async def example_tool_usage():
    agent = await get_agent()
    
    if agent:
        # The agent can reason about which tools to use
        response = await agent.run(
            "Check the current tags for microsoft/DialoGPT-medium and add the tag 'conversational-ai' if it's not already present"
        )
        print(response)



async def process_webhook_comment(webhook_data: Dict[str, Any]):
    """Process webhook to detect and add tags"""
    print("🏷️ Starting process_webhook_comment...")

    try:
        comment_content = webhook_data["comment"]["content"]
        discussion_title = webhook_data["discussion"]["title"]
        repo_name = webhook_data["repo"]["name"]
        
        # Extract potential tags from the comment and discussion title
        comment_tags = extract_tags_from_text(comment_content)
        title_tags = extract_tags_from_text(discussion_title)
        all_tags = list(set(comment_tags + title_tags))

        print(f"🔍 All unique tags: {all_tags}")

        if not all_tags:
            return ["No recognizable tags found in the discussion."]
        
                # Get agent instance
        agent = await get_agent()
        if not agent:
            return ["Error: Agent not configured (missing HF_TOKEN)"]

        # Process each tag
        result_messages = []
        for tag in all_tags:
            try:
                # Use agent to process the tag
                prompt = f"""
                For the repository '{repo_name}', check if the tag '{tag}' already exists.
                If it doesn't exist, add it via a pull request.
                
                Repository: {repo_name}
                Tag to check/add: {tag}
                """
                
                print(f"🤖 Processing tag '{tag}' for repo '{repo_name}'")
                response = await agent.run(prompt)
                
                # Parse agent response for success/failure
                if "success" in response.lower():
                    result_messages.append(f"✅ Tag '{tag}' processed successfully")
                else:
                    result_messages.append(f"⚠️ Issue with tag '{tag}': {response}")
                    
            except Exception as e:
                error_msg = f"❌ Error processing tag '{tag}': {str(e)}"
                print(error_msg)
                result_messages.append(error_msg)

        return result_messages
    
    
    


import re
from typing import List

# Recognized ML/AI tags for validation
RECOGNIZED_TAGS = {
    "pytorch", "tensorflow", "jax", "transformers", "diffusers",
    "text-generation", "text-classification", "question-answering",
    "text-to-image", "image-classification", "object-detection",
    "fill-mask", "token-classification", "translation", "summarization",
    "feature-extraction", "sentence-similarity", "zero-shot-classification",
    "image-to-text", "automatic-speech-recognition", "audio-classification",
    "voice-activity-detection", "depth-estimation", "image-segmentation",
    "video-classification", "reinforcement-learning", "tabular-classification",
    "tabular-regression", "time-series-forecasting", "graph-ml", "robotics",
    "computer-vision", "nlp", "cv", "multimodal",
}

def extract_tags_from_text(text: str) -> List[str]:
    """Extract potential tags from discussion text"""
    text_lower = text.lower()
    explicit_tags = []

    # Pattern 1: "tag: something" or "tags: something"
    tag_pattern = r"tags?:\s*([a-zA-Z0-9-_,\s]+)"
    matches = re.findall(tag_pattern, text_lower)
    for match in matches:
        tags = [tag.strip() for tag in match.split(",")]
        explicit_tags.extend(tags)

    # Pattern 2: "#hashtag" style
    hashtag_pattern = r"#([a-zA-Z0-9-_]+)"
    hashtag_matches = re.findall(hashtag_pattern, text_lower)
    explicit_tags.extend(hashtag_matches)

    # Pattern 3: Look for recognized tags mentioned in natural text
    mentioned_tags = []
    for tag in RECOGNIZED_TAGS:
        if tag in text_lower:
            mentioned_tags.append(tag)

    # Combine and deduplicate
    all_tags = list(set(explicit_tags + mentioned_tags))

    # Filter to only include recognized tags or explicitly mentioned ones
    valid_tags = []
    for tag in all_tags:
        if tag in RECOGNIZED_TAGS or tag in explicit_tags:
            valid_tags.append(tag)

    return valid_tags


from fastapi import BackgroundTasks

@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle webhook and process in background"""
    
    # Validate webhook quickly
    if request.headers.get("X-Webhook-Secret") != WEBHOOK_SECRET:
        return {"error": "Invalid secret"}
    
    webhook_data = await request.json()
    
    # Process in background to return quickly
    background_tasks.add_task(process_webhook_comment, webhook_data)
    
    return {"status": "accepted"}