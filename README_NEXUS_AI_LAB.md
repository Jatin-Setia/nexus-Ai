# NEXUS AI Lab Upgrade

This version adds an AI Lab to the existing NEXUS frontend/backend.

## Added
- AI Lab navigation section
- Gemini image generation endpoint: `/generate-image`
- Image Studio with style, aspect ratio and resolution controls
- Browser speech recognition for voice input
- Browser speech synthesis for speaking NEXUS responses
- Voice Lab recording + local preview effects
- Study Mode prompt builder
- AI Toolkit shortcuts for notes, quizzes, code coaching, presentations and interview practice

## Image generation setup
The backend uses the server-side `GEMINI_API_KEY` from `.env` and the Gemini image model `gemini-3.1-flash-image`.

Make sure the Google GenAI Python SDK is current enough to expose the image-generation API used by the project.

Example `.env` values:

```env
GEMINI_API_KEY=your_key_here
SECRET_KEY=your_secret_here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_password_here
```

## Run

```bash
source venv/bin/activate
python3 app.py
```

Open the NEXUS frontend using the same setup you already use for this project.

## Browser features
Voice input/output and microphone recording depend on browser support and permissions. Chrome/Edge are recommended for the speech-recognition feature.

The Voice Lab effects are client-side preview effects; they do not clone a real person's voice.
