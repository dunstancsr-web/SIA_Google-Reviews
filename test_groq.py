#!/usr/bin/env python3
"""Quick test to verify Groq API key works."""
import os
import sys

# Check if API key is set
api_key = os.environ.get("GROQ_API_KEY")
print(f"✓ API key in environment: {bool(api_key)}")
if api_key:
    print(f"  Key format: {api_key[:20]}...{api_key[-4:]}")
else:
    print("✗ GROQ_API_KEY not found in environment")
    print("  Please run: export GROQ_API_KEY='your_key_here'")
    sys.exit(1)

# Try to connect to Groq API
try:
    from groq import Groq
    print("✓ Groq library imported successfully")
    
    client = Groq(api_key=api_key)
    print("✓ Groq client created")
    
    # Make a simple test request
    message = client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Say 'hello' in one word."}
        ],
        model="mixtral-8x7b-32768",
        max_tokens=10,
    )
    response = message.choices[0].message.content.strip()
    print(f"✓ Groq API works! Response: '{response}'")
    
except ImportError as e:
    print(f"✗ Groq library not installed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Groq API error: {e}")
    sys.exit(1)
