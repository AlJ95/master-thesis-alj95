import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import openai

logger = logging.getLogger(__name__)

class LLMAsAJudge:
    """
    Utility class for making calls to LLMs for evaluation purposes.
    
    Uses the official OpenAI Python API client for API calls.
    """
    
    # Default OpenAI API URL
    DEFAULT_API_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize the LLM judge.
        
        Args:
            api_key: API key for the provider (uses env var OPENAI_API_KEY if None)
            api_url: Base URL for the API (uses OpenAI's URL if None)
            model: Model name to use
            temperature: Temperature for sampling (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the API call
        """
        # Set API key from input or environment variables
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and OPENAI_API_KEY not found in environment variables")
        
        # Set API URL (default to OpenAI)
        self.api_url = api_url or self.DEFAULT_API_URL
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
        
        # Set model and generation parameters
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = kwargs
    
    def evaluate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Make a call to the LLM with the provided prompt.
        
        Args:
            prompt: User prompt to send to the LLM
            system_prompt: Optional system prompt for chat models
            
        Returns:
            Tuple containing:
              - The generated text response
              - Raw API response as a dictionary
        """
        # Build messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Create parameters for the API call
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.additional_params
            }
            
            # Make API call using OpenAI client
            response = self.client.chat.completions.create(**params)
            
            # Extract content
            if not response.choices or len(response.choices) == 0:
                raise ValueError("Empty response from API")
                
            content = response.choices[0].message.content
            
            # Convert response to dictionary for consistency with previous implementation
            response_dict = response.model_dump()
            
            return content, response_dict
                
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ValueError(f"LLM API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ValueError(f"Unexpected error during LLM API request: {e}")
            
    def batch_evaluate(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Evaluate multiple prompts in batch.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (same for all prompts)
            
        Returns:
            List of (response, metadata) tuples
        """
        results = []
        
        for prompt in prompts:
            try:
                result = self.evaluate(prompt, system_prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch evaluation: {e}")
                # Append a failed result
                results.append(("", {"error": str(e)}))
                
        return results
        
    def judge_rating(
        self,
        prompt: str,
        expected_rating_format: str = "0-1 float",
        system_prompt: Optional[str] = None
    ) -> float:
        """
        Get a numerical rating from the LLM.
        
        Args:
            prompt: Prompt asking for a numerical rating
            expected_rating_format: Description of the expected format
            system_prompt: Optional system prompt
            
        Returns:
            Float rating value
        """
        if not system_prompt:
            system_prompt = f"""You are an expert evaluator. 
Your task is to provide an objective rating based on the criteria given.
Your response must be ONLY a number in the format: {expected_rating_format}.
Do not include any explanations or other text in your response."""
            
        content, _ = self.evaluate(prompt, system_prompt)
        
        # Try to parse a float from the response
        try:
            # Clean the response and extract the float
            cleaned = content.strip().split("\n")[0].strip()
            
            # Handle specific formats
            if expected_rating_format == "0-1 float":
                rating = float(cleaned)
                # Ensure it's in the right range
                rating = max(0.0, min(1.0, rating))
                return rating
            elif expected_rating_format == "1-5 int":
                rating = int(cleaned)
                # Ensure it's in the right range
                rating = max(1, min(5, rating))
                return float(rating) / 5.0  # Normalize to 0-1
            else:
                # Default: just parse as float
                return float(cleaned)
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse rating from '{content}': {e}")
            # Default to middle value
            return 0.5 