# Sub700ms AI Poetry API

A FastAPI backend that generates poetry in under 700ms. Features real-time streaming and performance tracking.

**Live Demo:** https://sub700.onrender.com

## Setup

```bash
git clone https://github.com/your-username/sub700.git
cd sub700
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key" > .env
python main.py
```

## Endpoints

- `POST /chat/complete` - Get a poem with timing data
- `POST /chat/stream` - Watch the poem generate word by word
- `GET /metrics/latency` - See performance stats

## Testing Performance

**Quick test for speed:**
```http
POST https://sub700.onrender.com/chat/complete
Content-Type: application/json

{
  "message": "Write one word",
  "max_tokens": 10
}
```

Look for `total_time_ms` in the response - should be under 700ms.

**Streaming test:**
```http
POST https://sub700.onrender.com/chat/stream
Content-Type: application/json

{
  "message": "Write a haiku about speed"
}
```

You'll see the poem appear word by word in real-time.

**Check stats:**
```http
GET https://sub700.onrender.com/metrics/latency
```

The `sub_700ms_percentage` tells you how often requests finish under 700ms.

## Understanding the Numbers

- `total_time_ms` - How long the whole request took
- `time_to_first_token_ms` - How fast streaming starts  
- `sub_700ms_percentage` - What percent of requests are fast enough

The first request is always slower because connections need to warm up. Try a few requests to see the real performance.

## Deploy Your Own

1. Fork this repo
2. Connect to Render
3. Set `OPENAI_API_KEY` environment variable
4. Deploy

That's it. No config needed - everything's in the code.
