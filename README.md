# Sub-700ms AI Poetry API 🚀

FastAPI backend optimized for **sub-700ms latency** with streaming poetry generation and real-time performance tracking.

## 🎯 Performance Goal

**Target: Sub-700ms response times** with comprehensive latency monitoring.

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/sub700.git
cd sub700
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key-here" > .env
python main.py
```

## 📡 Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /chat/complete` | Non-streaming poetry (shows timing) |
| `POST /chat/stream` | Real-time streaming poetry |
| `GET /metrics/latency` | Performance statistics |

## 🧪 Testing Latency in Postman

### 1. Test Sub-700ms Performance

```http
POST http://localhost:8000/chat/complete
Content-Type: application/json

{
  "message": "Write one word",
  "max_tokens": 10
}
```

**Response shows timing:**
```json
{
  "response": "Creation",
  "timing": {
    "total_time_ms": 586.27,     // ← Should be under 700ms
    "llm_time_ms": 586.27,
    "processing_time_ms": 0.0
  }
}
```

### 2. Test Streaming (Real-time)

```http
POST http://localhost:8000/chat/stream
Content-Type: application/json

{
  "message": "Write a haiku about speed"
}
```

**Streaming response:**
```
data: {"type": "timing", "time_to_first_token_ms": 145.67}
data: {"type": "content", "content": "Swift"}
data: {"type": "content", "content": " wind"}
data: {"type": "content", "content": " flows"}
data: {"type": "complete", "timing": {"total_time_ms": 850.45}}
data: [DONE]
```

### 3. Check Performance Stats

```http
GET http://localhost:8000/metrics/latency
```

**Key metrics:**
```json
{
  "latency_ms": {
    "mean": 612.34,
    "p95": 756.89,
    "p99": 789.45
  },
  "sub_700ms_percentage": 85.0   // ← Target: >90%
}
```

## 📊 Understanding Latency

- **`total_time_ms`**: Complete request time (optimization target)
- **`time_to_first_token_ms`**: How fast streaming starts
- **`sub_700ms_percentage`**: % of requests under 700ms
- **Look at Postman's timing** in bottom-right of response

## 🚀 Deploy to Render

1. Push to GitHub
2. Connect to Render as Web Service  
3. Set environment variable: `OPENAI_API_KEY`
4. Deploy automatically

**Built for speed. Optimized for real-time AI.** ⚡