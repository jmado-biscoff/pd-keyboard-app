---
description: Google Vision API OCR integration for deployed environments
applyTo: "**/ocrRoutes.ts"
---

# Google Vision API OCR Integration

## Overview
This codebase includes Google Vision API integration for OCR (Optical Character Recognition) scanning functionality. The API is used to extract text from images captured during typing sessions.

## Implementation Details

### Authentication Methods (Priority Order)
1. **Service Account JSON** (`GOOGLE_CREDENTIALS_JSON` env var) - Uses `@google-cloud/vision` client library
2. **API Key** (`GOOGLE_VISION_API_KEY` env var) - Uses REST API with axios
3. **Local Service Account File** - Falls back to local JSON file for development

### Key Architecture Decision
The `@google-cloud/vision` Node.js client library **does NOT support API key authentication**. It only works with service account credentials. Therefore:
- For service accounts: We use the official client library
- For API keys: We call the REST API directly using axios

## Environment Setup

### For Deployed Environments
Set **one** of these environment variables:

```bash
# Option 1: Service Account JSON (Recommended - more secure)
GOOGLE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project",...}

# Option 2: API Key (Simpler setup, works with REST API)
GOOGLE_VISION_API_KEY=your_api_key_here
```

### For Local Development
1. Place your service account JSON file as `expomapapp-475414-c637430cbe92.json` in the backend directory
2. Or set `GOOGLE_CREDENTIALS_JSON` or `GOOGLE_VISION_API_KEY` in your `.env` file

### Getting Google Vision API Credentials

**Service Account (Recommended):**
1. Go to Google Cloud Console > IAM & Admin > Service Accounts
2. Create a service account with "Cloud Vision API User" role
3. Create a JSON key and download it
4. Set the JSON content as `GOOGLE_CREDENTIALS_JSON` environment variable

**API Key (Alternative):**
1. Go to Google Cloud Console > APIs & Services > Credentials
2. Create an API key
3. Enable Vision API for your project
4. Set the key as `GOOGLE_VISION_API_KEY` environment variable
5. Optionally restrict the key to only Vision API

## API Endpoint

### POST /api/ocr/parse

**Request Body:**
```json
{
  "base64Image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "ParsedResults": [
    {
      "ParsedText": "extracted text here"
    }
  ]
}
```

**Error Response:**
```json
{
  "error": "Error description",
  "details": "Additional details (dev mode only)",
  "ParsedResults": [
    {
      "ParsedText": "",
      "ErrorMessage": "Error message"
    }
  ]
}
```

## Testing

### Manual Testing with curl
```bash
# Test with a small base64 image
curl -X POST http://localhost:5000/api/ocr/parse \
  -H "Content-Type: application/json" \
  -d '{"base64Image":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}'
```

### Check Authentication Method
The server logs will show which authentication method is being used:
- `✅ OCR Auth: Using service account from GOOGLE_CREDENTIALS_JSON`
- `✅ OCR Auth: Using API key from GOOGLE_VISION_API_KEY`
- `✅ OCR Auth: Using local service account file: ...`
- `⚠️ OCR Auth: No Google Vision credentials found`

## Common Issues

### 1. "Google Vision API not configured"
**Problem:** Neither `GOOGLE_CREDENTIALS_JSON` nor `GOOGLE_VISION_API_KEY` is set
**Solution:** Set one of these environment variables

### 2. "PERMISSION_DENIED" or 403 errors
**Problem:** Credentials don't have Vision API access
**Solution:**
- For service accounts: Add "Cloud Vision API User" role
- For API keys: Ensure Vision API is enabled and key is valid

### 3. "RESOURCE_EXHAUSTED" or 429 errors
**Problem:** API quota exceeded
**Solution:** Wait and retry, or upgrade Google Cloud quota

### 4. API Key not working with client library
**Problem:** The `@google-cloud/vision` library doesn't support API keys
**Solution:** This implementation automatically uses REST API for API key auth

### 5. Invalid JSON parsing for GOOGLE_CREDENTIALS_JSON
**Problem:** Environment variable contains malformed JSON
**Solution:** Ensure proper escaping:
```bash
# Correct format (single quotes, escaped quotes inside)
GOOGLE_CREDENTIALS_JSON='{"type":"service_account",...}'
```

## Deployment Checklist
- [ ] Set `GOOGLE_CREDENTIALS_JSON` or `GOOGLE_VISION_API_KEY` in deployment platform
- [ ] Verify service account has Vision API permissions (if using service account)
- [ ] Enable Vision API in Google Cloud project
- [ ] Test OCR endpoint after deployment
- [ ] Monitor API usage and costs in Google Cloud Console