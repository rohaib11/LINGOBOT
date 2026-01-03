# server.py - Complete Enhanced Version
import os
import io
import base64
import uuid
import json
import time
import asyncio
import hashlib
import difflib
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

# FASTAPI IMPORTS
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# RATE LIMITING IMPORTS
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# AI & DATA IMPORTS
from pydantic import BaseModel, Field, validator
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import whisper
from deep_translator import GoogleTranslator
import redis.asyncio as redis
import jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import logging
from dataclasses import dataclass
from enum import Enum
import psutil
from fastapi import Response # Add this
import magic
from pydub import AudioSegment
import structlog

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
except:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# 1. INITIALIZATION
load_dotenv()

# Setup Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="LingoBot Pro API",
    description="Advanced AI Language Learning Platform",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Connect Rate Limiter to App
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enhanced CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Verify API Key exists
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    logger.warning("GROQ_API_KEY not found in environment variables. Some features may not work.")

try:
    client = Groq(api_key=api_key)
except:
    client = None
    logger.warning("Failed to initialize Groq client")

# 2. DATABASE & CACHE SETUP
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/lingobot.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis for caching and real-time features
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_client = redis.from_url(redis_url, decode_responses=True)
except:
    logger.warning(f"Failed to connect to Redis at {redis_url}")
    redis_client = None

# 3. MODELS
class ChatMode(str, Enum):
    TUTOR = "tutor"
    EXAMINER = "examiner"
    TRANSLATOR = "translator"
    DEBATE = "debate"
    ROLEPLAY = "roleplay"
    CORRECTOR = "corrector"

class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default=UserRole.STUDENT.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    settings = Column(JSON, default={
        "default_language": "en",
        "notifications": True,
        "theme": "light",
        "audio_enabled": True,
        "difficulty": "intermediate"
    })
    progress = Column(JSON, default={
        "level": 1,
        "xp": 0,
        "streak": 0,
        "total_sessions": 0,
        "total_time": 0,
        "skills": {
            "grammar": {"score": 0, "level": 1},
            "vocabulary": {"score": 0, "level": 1},
            "pronunciation": {"score": 0, "level": 1},
            "fluency": {"score": 0, "level": 1},
            "comprehension": {"score": 0, "level": 1}
        },
        "achievements": []
    })

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    messages = Column(JSON)
    language = Column(String, default="en")
    mode = Column(String, default=ChatMode.TUTOR.value)
    difficulty = Column(String, default=DifficultyLevel.INTERMEDIATE.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    analytics = Column(JSON, default={})
    duration = Column(Integer, default=0)  # in seconds

class LearningMaterial(Base):
    __tablename__ = "learning_materials"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String)
    description = Column(Text)
    content = Column(Text)
    language = Column(String)
    difficulty = Column(String)
    category = Column(String)
    tags = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    usage_count = Column(Integer, default=0)

class Achievement(Base):
    __tablename__ = "achievements"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    description = Column(Text)
    icon = Column(String)
    criteria = Column(JSON)
    xp_reward = Column(Integer, default=100)

Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, description="Password")

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    username: str
    role: str
    expires_in: int

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_text: str
    language: str = "en"
    mode: ChatMode = ChatMode.TUTOR
    context: Optional[Dict] = None
    emotion: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE

class AudioRequest(BaseModel):
    audio_base64: str
    language: str = "en"
    reference_text: Optional[str] = None
    calculate_score: bool = True

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"
    include_cultural_notes: bool = True
    include_alternatives: bool = True

class ProgressUpdate(BaseModel):
    skill: str
    level: int
    score: float
    xp_gained: Optional[int] = 0

class LearningPathRequest(BaseModel):
    language: str
    level: str
    focus_areas: Optional[List[str]] = None
    time_per_day: int = 30  # minutes

class GrammarAnalysisRequest(BaseModel):
    text: str
    language: str = "en"
    detailed: bool = False

# 4. AUTHENTICATION
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return user_id

# 5. AI SERVICES

class AIService:
    def __init__(self):
        self.whisper_model = None
        self.embedding_model = None
        self.sentiment_analyzer = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_lock = asyncio.Lock()
        
    async def load_models(self):
        """Load all ML models asynchronously"""
        logger.info("Starting model loading...")
        
        # Load models in parallel
        tasks = [
            self._load_whisper(),
            self._load_embedding_model(),
            self._load_sentiment_analyzer()
        ]
        
        await asyncio.gather(*tasks)
        logger.info("All models loaded successfully")

    async def _load_whisper(self):
        async with self.model_lock:
            if self.whisper_model is None:
                logger.info("Loading Whisper model...")
                whisper_size = os.getenv("WHISPER_MODEL_SIZE", "base")
                self.whisper_model = whisper.load_model(whisper_size)
                logger.info(f"Whisper model ({whisper_size}) loaded")

    async def _load_embedding_model(self):
        async with self.model_lock:
            if self.embedding_model is None:
                logger.info("Loading Embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded")

    async def _load_sentiment_analyzer(self):
        async with self.model_lock:
            if self.sentiment_analyzer is None:
                logger.info("Loading Sentiment Analyzer...")
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("Sentiment Analyzer loaded")

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        await self._load_whisper()
        
        def _transcribe():
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                # Convert to wav if needed
                try:
                    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(tmp.name, format="wav")
                except:
                    tmp.write(audio_bytes)
                
                tmp.flush()
                return self.whisper_model.transcribe(tmp.name, language="en")["text"]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _transcribe)

    async def get_embedding(self, text: str):
        await self._load_embedding_model()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.embedding_model.encode(text)
        )

    def analyze_sentiment(self, text: str) -> Dict:
        if self.sentiment_analyzer is None:
            return {"sentiment": "neutral", "scores": {"compound": 0, "pos": 0, "neu": 0, "neg": 0}}
        
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "scores": scores,
            "intensity": abs(compound)
        }
    
    def analyze_grammar(self, text: str, detailed: bool = False) -> Dict:
        import re
        import string
        
        issues = []
        suggestions = []
        score = 100
        
        # Basic checks
        words = text.split()
        word_count = len(words)
        
        if word_count < 3:
            issues.append("Sentence may be too short for meaningful practice")
            score -= 10
        
        # Check for multiple spaces
        if '  ' in text:
            issues.append("Multiple consecutive spaces detected")
            score -= 5
        
        # Check capitalization
        sentences = re.split(r'[.!?]+', text)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and sentence[0].islower():
                issues.append(f"Sentence {i+1} doesn't start with capital letter")
                score -= 5
        
        # Punctuation check
        if text and text[-1] not in '.!?':
            issues.append("Sentence may be missing ending punctuation")
            score -= 5
        
        # Very overuse check
        very_count = len(re.findall(r'\bvery\b', text, re.IGNORECASE))
        if very_count > 2:
            issues.append(f"Overuse of 'very' ({very_count} times) - consider stronger adjectives")
            score -= min(very_count * 2, 10)
            suggestions.append("Try using more descriptive adjectives instead of 'very + adjective'")
        
        # Repetition check
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip(string.punctuation)
            if len(word_lower) > 3:
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        repeated_words = [word for word, count in word_freq.items() if count > 3]
        if repeated_words:
            issues.append(f"Repeated words: {', '.join(repeated_words[:3])}")
            score -= len(repeated_words) * 3
            suggestions.append("Try using synonyms for repeated words")
        
        if detailed:
            # Additional detailed analysis
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            return {
                "issues": issues,
                "suggestions": suggestions,
                "score": max(0, score),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "repeated_words": repeated_words,
                "very_count": very_count
            }
        else:
            return {
                "issues": issues,
                "suggestions": suggestions[:2],
                "score": max(0, score),
                "word_count": word_count
            }

    def calculate_pronunciation_score(self, reference_text: str, spoken_text: str) -> Dict:
        import re
        from collections import Counter
        
        def clean(t):
            return re.sub(r'[^\w\s]', '', t).lower().strip()
        
        def tokenize(t):
            return re.findall(r'\b\w+\b', t.lower())
        
        ref_clean = clean(reference_text)
        hyp_clean = clean(spoken_text)
        
        if not ref_clean:
            return {
                "score": 0,
                "similarity": 0,
                "word_accuracy": 0,
                "missing_words": [],
                "extra_words": [],
                "confidence": 0
            }
        
        # Calculate similarity using SequenceMatcher
        matcher = difflib.SequenceMatcher(None, ref_clean, hyp_clean)
        similarity = matcher.ratio()
        
        # Tokenize and compare
        ref_tokens = tokenize(reference_text)
        hyp_tokens = tokenize(spoken_text)
        
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)
        
        correct_words = ref_set.intersection(hyp_set)
        missing_words = ref_set - hyp_set
        extra_words = hyp_set - ref_set
        
        word_accuracy = len(correct_words) / len(ref_set) if ref_set else 0
        
        # Calculate final score (weighted)
        similarity_weight = 0.4
        word_accuracy_weight = 0.6
        
        final_score = int((similarity * similarity_weight + word_accuracy * word_accuracy_weight) * 100)
        
        # Adjust for length difference
        length_diff = abs(len(ref_tokens) - len(hyp_tokens))
        if length_diff > 3:
            final_score = max(0, final_score - (length_diff * 5))
        
        return {
            "score": final_score,
            "similarity": round(similarity * 100, 1),
            "word_accuracy": round(word_accuracy * 100, 1),
            "missing_words": list(missing_words)[:5],
            "extra_words": list(extra_words)[:5],
            "confidence": min(final_score / 100, 1.0)
        }

    def extract_vocabulary(self, text: str) -> List[Dict]:
        """Extract potential vocabulary words from text"""
        import re
        from nltk.corpus import stopwords
        
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter out stopwords and get unique words
        vocab_words = []
        seen = set()
        
        for word in words:
            if word not in stop_words and word not in seen:
                seen.add(word)
                # Simple difficulty estimation
                difficulty = "easy" if len(word) <= 5 else "medium" if len(word) <= 8 else "hard"
                vocab_words.append({
                    "word": word,
                    "difficulty": difficulty,
                    "length": len(word),
                    "frequency": words.count(word)
                })
        
        return sorted(vocab_words, key=lambda x: x["frequency"], reverse=True)[:10]

ai_service = AIService()

# 6. WEBSOCKET MANAGER
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from user sessions
        user_id = None
        for uid, sid in self.user_sessions.items():
            if sid == client_id:
                user_id = uid
                break
        
        if user_id:
            del self.user_sessions[user_id]
            
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)
    
    async def send_to_user(self, user_id: str, message: Dict):
        """Send message to specific user"""
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            await self.send_personal_message(json.dumps(message), session_id)

manager = ConnectionManager()

# 7. ENHANCED SYSTEM PROMPTS
def get_system_prompt(mode: ChatMode, language: str, difficulty: DifficultyLevel, context: Optional[Dict] = None) -> str:
    base_prompts = {
        ChatMode.TUTOR: f"""
        You are an expert {language} tutor with 20 years of experience.
        Teaching style: patient, encouraging, and precise.
        Difficulty level: {difficulty.value}
        
        GUIDELINES:
        1. Keep responses conversational but educational
        2. Provide subtle corrections within context
        3. Include cultural insights when relevant
        4. End with ONE follow-up question to continue conversation
        5. Use emojis occasionally for warmth 😊
        6. Break complex concepts into digestible parts
        7. Adjust vocabulary based on difficulty level
        
        User context: {context or 'No prior context'}
        
        Current focus: Improving {language} fluency through natural conversation.
        """,
        
        ChatMode.EXAMINER: f"""
        You are a formal {language} examination board certified examiner.
        Your task: Conduct a simulated language proficiency test.
        Difficulty level: {difficulty.value}
        
        EXAMINER PROTOCOL:
        1. Ask ONE question at a time
        2. Assess: Pronunciation, Vocabulary, Grammar, Fluency, Coherence
        3. Provide numerical score (0-9 band) after each response
        4. Note specific improvements needed
        5. Maintain professional, neutral tone
        6. Adjust question difficulty based on performance
        
        Assessment in progress...
        """,
        
        ChatMode.TRANSLATOR: f"""
        You are a master translator and cultural bridge between languages.
        
        TRANSLATION PROTOCOL:
        1. Provide accurate translation
        2. Explain cultural nuances
        3. Highlight idioms and their equivalents
        4. Note formal/informal registers
        5. Suggest alternative expressions
        6. Point out common translation mistakes
        
        Current language pair: {language} ↔ English
        Difficulty level: {difficulty.value}
        Focus on natural, idiomatic expressions.
        """,
        
        ChatMode.DEBATE: f"""
        You are a debate partner for {language} practice.
        Difficulty level: {difficulty.value}
        
        DEBATE RULES:
        1. Take a position on given topic
        2. Use formal debate language
        3. Present clear arguments
        4. Challenge user's points constructively
        5. Use transition phrases for coherence
        6. Help user improve argumentation skills
        
        Goal: Improve argumentation and persuasion skills in {language}.
        """,
        
        ChatMode.ROLEPLAY: f"""
        You are a role-play partner for {language} immersion.
        Difficulty level: {difficulty.value}
        
        SCENARIO: {context.get('scenario', 'Casual conversation in a café')}
        
        ROLE-PLAY GUIDELINES:
        1. Stay in character
        2. Use appropriate register (formal/informal)
        3. Introduce natural dialogue elements
        4. Create opportunities for specific vocabulary use
        5. Make it fun and engaging!
        6. Provide subtle language guidance when needed
        """,
        
        ChatMode.CORRECTOR: f"""
        You are a meticulous {language} grammar and style corrector.
        Difficulty level: {difficulty.value}
        
        CORRECTION PROTOCOL:
        1. First, provide the corrected version
        2. Explain each correction clearly
        3. Suggest alternative phrasings
        4. Highlight common mistakes
        5. Provide memory tips for rules
        6. Be encouraging and constructive
        
        Focus on: Grammar, syntax, word choice, style, and naturalness.
        """
    }
    
    return base_prompts.get(mode, base_prompts[ChatMode.TUTOR])

# 8. CACHE MANAGEMENT
async def get_cached_audio(text: str, lang: str) -> Optional[str]:
    """Retrieve audio from cache if available"""
    if not redis_client:
        return None
    
    try:
        clean_text = text.strip().lower()
        key = f"tts:{lang}:{hashlib.md5(clean_text.encode()).hexdigest()}"
        cached_audio = await redis_client.get(key)
        if cached_audio:
            logger.info("Audio cache hit", text_length=len(text), language=lang)
            return cached_audio
    except Exception as e:
        logger.error("Cache get error", error=str(e))
    return None

async def cache_audio(text: str, lang: str, audio_base64: str):
    """Save generated audio to cache"""
    if not redis_client:
        return
    
    try:
        clean_text = text.strip().lower()
        key = f"tts:{lang}:{hashlib.md5(clean_text.encode()).hexdigest()}"
        ttl = int(os.getenv("AUDIO_CACHE_TTL", 86400))
        await redis_client.setex(key, ttl, audio_base64)
        logger.info("Audio cached", text_length=len(text), language=lang, ttl=ttl)
    except Exception as e:
        logger.error("Cache set error", error=str(e))

async def get_cached_response(prompt_hash: str) -> Optional[Dict]:
    """Cache AI responses for common prompts"""
    if not redis_client:
        return None
    
    try:
        cached = await redis_client.get(f"response:{prompt_hash}")
        if cached:
            return json.loads(cached)
    except:
        pass
    return None

async def cache_response(prompt_hash: str, response: Dict, ttl: int = 3600):
    """Cache AI response"""
    if not redis_client:
        return
    
    try:
        await redis_client.setex(f"response:{prompt_hash}", ttl, json.dumps(response))
    except:
        pass

# 9. API ENDPOINTS

@app.post("/auth/register", response_model=Token)
@limiter.limit("10/minute")
async def register(request: Request, user: UserCreate):
    """Register a new user"""
    db = SessionLocal()
    try:
        # Check if user exists
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        existing_username = db.query(User).filter(User.username == user.username).first()
        if existing_username:
            raise HTTPException(status_code=400, detail="Username already taken")
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        db_user = User(
            email=user.email,
            username=user.username,
            hashed_password=hashed_password,
            settings={
                "default_language": "en",
                "notifications": True,
                "theme": "light",
                "audio_enabled": True,
                "difficulty": "intermediate"
            },
            progress={
                "level": 1,
                "xp": 0,
                "streak": 0,
                "total_sessions": 0,
                "total_time": 0,
                "skills": {
                    "grammar": {"score": 0, "level": 1},
                    "vocabulary": {"score": 0, "level": 1},
                    "pronunciation": {"score": 0, "level": 1},
                    "fluency": {"score": 0, "level": 1},
                    "comprehension": {"score": 0, "level": 1}
                },
                "achievements": []
            }
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # Create token
        access_token = create_access_token(
            data={"sub": db_user.id, "role": db_user.role}
        )
        
        # Initialize user stats in Redis
        if redis_client:
            await redis_client.hset(f"user:{db_user.id}:stats", "session_count", 0)
            await redis_client.hset(f"user:{db_user.id}:stats", "streak", 0)
            await redis_client.hset(f"user:{db_user.id}:stats", "last_active", datetime.utcnow().isoformat())
        
        logger.info("User registered", user_id=db_user.id, email=user.email)
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=db_user.id,
            username=db_user.username,
            role=db_user.role,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        db.close()

@app.post("/auth/login", response_model=Token)
@limiter.limit("20/minute")
async def login(request: Request, user: UserLogin):
    """Login user"""
    db = SessionLocal()
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update last login
        db_user.last_login = datetime.utcnow()
        db.commit()
        
        # Create token
        access_token = create_access_token(
            data={"sub": db_user.id, "role": db_user.role}
        )
        
        # Update Redis stats
        if redis_client:
            today = datetime.utcnow().date().isoformat()
            last_active = await redis_client.hget(f"user:{db_user.id}:stats", "last_active")
            
            if last_active != today:
                current_streak = int(await redis_client.hget(f"user:{db_user.id}:stats", "streak") or 0)
                if last_active and (datetime.fromisoformat(last_active).date() == 
                                   datetime.utcnow().date() - timedelta(days=1)):
                    current_streak += 1
                else:
                    current_streak = 1
                
                await redis_client.hset(f"user:{db_user.id}:stats", "streak", current_streak)
                await redis_client.hset(f"user:{db_user.id}:stats", "last_active", today)
        
        logger.info("User logged in", user_id=db_user.id, email=user.email)
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=db_user.id,
            username=db_user.username,
            role=db_user.role,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        db.close()

@app.post("/chat/advanced", response_model=Dict)
@limiter.limit("30/minute")
async def advanced_chat(
    request: Request,
    chat_data: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Enhanced chat endpoint with analytics, caching, and rate limiting"""
    session_id = chat_data.session_id or str(uuid.uuid4())
    user_id = current_user or "anonymous"
    
    try:
        # 1. Analyze user input
        sentiment = ai_service.analyze_sentiment(chat_data.user_text)
        grammar = ai_service.analyze_grammar(chat_data.user_text, detailed=True)
        vocabulary = ai_service.extract_vocabulary(chat_data.user_text)
        
        # 2. Prepare context
        enhanced_context = {
            "user_id": user_id,
            "sentiment": sentiment,
            "grammar_analysis": grammar,
            "vocabulary": vocabulary[:5],
            "timestamp": datetime.utcnow().isoformat(),
            "difficulty": chat_data.difficulty.value,
            "emotion": chat_data.emotion,
            **(chat_data.context or {})
        }
        
        # 3. Check cache for similar prompts
        prompt_hash = hashlib.md5(
            f"{chat_data.user_text}:{chat_data.language}:{chat_data.mode.value}:{chat_data.difficulty.value}".encode()
        ).hexdigest()
        
        cached_response = await get_cached_response(prompt_hash)
        if cached_response:
            logger.info("Response cache hit", prompt_hash=prompt_hash[:8])
            # Update context with current data
            cached_response["enhanced_context"] = enhanced_context
            cached_response["session_id"] = session_id
            
            # Background tasks for analytics
            background_tasks.add_task(
                store_conversation_async, 
                session_id, user_id, chat_data, cached_response
            )
            if current_user:
                background_tasks.add_task(
                    update_user_progress_async, 
                    user_id, chat_data, cached_response
                )
            
            return cached_response
        
        # 4. Generate AI response
        system_prompt = get_system_prompt(
            chat_data.mode, 
            chat_data.language, 
            chat_data.difficulty, 
            enhanced_context
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_data.user_text}
        ]
        
        # Add conversation history if available
        if chat_data.context and "history" in chat_data.context:
            messages = chat_data.context["history"][-5:] + messages  # Last 5 messages
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile" if chat_data.difficulty == DifficultyLevel.ADVANCED else "llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=False
            )
            
            ai_reply = response.choices[0].message.content
            learning_points = extract_learning_points(ai_reply, chat_data.user_text)
            
        except Exception as e:
            logger.error("Groq API error", error=str(e))
            # Fallback response
            ai_reply = "I apologize, but I'm having trouble processing your request. Please try again or rephrase your question."
            learning_points = []
        
        # 5. Generate TTS (With Smart Caching)
        audio_base64 = await get_cached_audio(ai_reply, chat_data.language)
        
        if not audio_base64:
            def generate_audio():
                try:
                    mp3_fp = io.BytesIO()
                    tts = gTTS(text=ai_reply, lang=chat_data.language, slow=False)
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)
                    return base64.b64encode(mp3_fp.read()).decode('utf-8')
                except Exception as e:
                    logger.error("TTS generation error", error=str(e))
                    return None
            
            loop = asyncio.get_event_loop()
            audio_base64 = await loop.run_in_executor(None, generate_audio)
            
            if audio_base64:
                background_tasks.add_task(cache_audio, ai_reply, chat_data.language, audio_base64)
        
        # 6. Prepare Response
        response_data = {
            "session_id": session_id,
            "ai_reply": ai_reply,
            "learning_points": learning_points,
            "vocabulary_highlight": vocabulary[:5],
            "sentiment_feedback": sentiment,
            "grammar_feedback": grammar,
            "audio_base64": audio_base64,
            "enhanced_context": enhanced_context,
            "suggested_next_topics": suggest_topics(chat_data.user_text),
            "mode": chat_data.mode.value,
            "language": chat_data.language,
            "difficulty": chat_data.difficulty.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 7. Cache the response
        background_tasks.add_task(cache_response, prompt_hash, response_data)
        
        # 8. Background analytics tasks
        background_tasks.add_task(
            store_conversation_async, 
            session_id, user_id, chat_data, response_data
        )
        
        if current_user:
            background_tasks.add_task(
                update_user_progress_async, 
                user_id, chat_data, response_data
            )
        
        logger.info("Chat request processed", 
                   user_id=user_id, 
                   session_id=session_id[:8],
                   mode=chat_data.mode.value,
                   language=chat_data.language)
        
        return response_data
        
    except Exception as e:
        logger.error("Chat error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/audio/transcribe")
@limiter.limit("20/minute")
async def transcribe_audio(request: Request, audio_data: AudioRequest):
    """Transcribe audio to text with pronunciation scoring"""
    try:
        # Check audio size
        audio_size = len(audio_data.audio_base64) * 3 / 4  # Base64 approximation
        max_size = int(os.getenv("MAX_AUDIO_SIZE_MB", 10)) * 1024 * 1024
        
        if audio_size > max_size:
            raise HTTPException(status_code=413, detail=f"Audio file too large. Maximum size is {max_size/1024/1024}MB")
        
        audio_bytes = base64.b64decode(audio_data.audio_base64)
        
        # Transcribe
        transcription = await ai_service.transcribe_audio(audio_bytes)
        
        # Analyze
        sentiment = ai_service.analyze_sentiment(transcription)
        grammar = ai_service.analyze_grammar(transcription)
        vocabulary = ai_service.extract_vocabulary(transcription)
        
        # Calculate Pronunciation Score
        pronunciation_data = {"score": 0, "similarity": 0, "word_accuracy": 0}
        if audio_data.reference_text and audio_data.calculate_score: # <--- FIXED
            pronunciation_data = ai_service.calculate_pronunciation_score(
                audio_data.reference_text, 
                transcription
            )
        
        return {
            "text": transcription,
            "language": audio_data.language,
            "sentiment": sentiment,
            "grammar_score": grammar["score"],
            "grammar_issues": grammar.get("issues", []),
            "vocabulary": vocabulary[:3],
            "pronunciation": pronunciation_data,
            "confidence": 0.95,
            "word_count": len(transcription.split()),
            "processing_time": "instant"  # Would be actual time in production
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Audio transcription error", error=str(e))
        raise HTTPException(status_code=500, detail="Audio processing failed")

@app.post("/translate/advanced")
@limiter.limit("30/minute")
async def advanced_translation(request: Request, translation_request: TranslationRequest):
    """Enhanced translation with cultural context"""
    try:
        # Translate
        translator = GoogleTranslator(
            source=request.source_lang,
            target=request.target_lang
        )
        translated = translator.translate(request.text)
        
        response_data = {
            "original": request.text,
            "translated": translated,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "difficulty_level": assess_translation_difficulty(request.text),
            "word_count": len(request.text.split()),
            "character_count": len(request.text)
        }
        
        # Get cultural notes if requested
        if request.include_cultural_notes:
            cultural_notes = await get_cultural_notes(request.text, request.target_lang)
            response_data["cultural_notes"] = cultural_notes
        
        # Get alternative translations if requested
        if request.include_alternatives:
            alternatives = await get_translation_alternatives(request.text, request.target_lang)
            response_data["alternatives"] = alternatives
        
        return response_data
    except Exception as e:
        logger.error("Translation error", error=str(e))
        raise HTTPException(status_code=500, detail="Translation service unavailable")

@app.post("/grammar/analyze")
@limiter.limit("30/minute")
async def analyze_grammar(request: GrammarAnalysisRequest):
    """Advanced grammar analysis"""
    try:
        analysis = ai_service.analyze_grammar(request.text, detailed=request.detailed)
        
        # Add language-specific suggestions
        if grammar_data.language == "en": # <--- FIXED
            analysis["language_specific_tips"] = [
                "Remember to use articles (a, an, the) appropriately",
                "Check subject-verb agreement",
                "Use correct prepositions for context"
            ]
        
        return analysis
    except Exception as e:
        logger.error("Grammar analysis error", error=str(e))
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/learning/path/{language}/{level}")
async def get_learning_path(
    language: str, 
    level: str,
    focus_areas: Optional[str] = None,
    time_per_day: int = 30
):
    """Get personalized learning path"""
    focus_list = focus_areas.split(",") if focus_areas else ["conversation", "vocabulary", "grammar"]
    
    paths = {
        "beginner": {
            "title": f"{language.title()} Beginner Journey",
            "description": "Master the basics and build confidence",
            "weeks": 8,
            "daily_minutes": time_per_day,
            "focus_areas": focus_list,
            "total_hours": (8 * 7 * time_per_day) / 60,
            "milestones": [
                {"week": 1, "focus": "Greetings & Introductions", "goals": ["Introduce yourself", "Ask basic questions"]},
                {"week": 2, "focus": "Common Phrases", "goals": ["Order food", "Ask for directions"]},
                {"week": 3, "focus": "Present Tense", "goals": ["Describe daily routine", "Talk about hobbies"]},
                {"week": 4, "focus": "Daily Routines", "goals": ["Describe your day", "Talk about work/school"]},
                {"week": 5, "focus": "Food & Ordering", "goals": ["Read a menu", "Order at restaurant"]},
                {"week": 6, "focus": "Directions & Transportation", "goals": ["Ask for directions", "Use public transport"]},
                {"week": 7, "focus": "Shopping & Numbers", "goals": ["Ask for prices", "Make purchases"]},
                {"week": 8, "focus": "Review & Cultural Insights", "goals": ["Consolidate learning", "Learn cultural norms"]}
            ],
            "resources": [
                {"type": "vocabulary", "count": 200, "description": "Essential words"},
                {"type": "phrases", "count": 50, "description": "Common expressions"},
                {"type": "grammar", "count": 10, "description": "Basic rules"}
            ]
        },
        "intermediate": {
            "title": f"{language.title()} Intermediate Mastery",
            "description": "Build fluency and expand vocabulary",
            "weeks": 12,
            "daily_minutes": time_per_day,
            "focus_areas": focus_list,
            "total_hours": (12 * 7 * time_per_day) / 60,
            "milestones": [
                {"week": 1, "focus": "Past Tense", "goals": ["Tell stories", "Describe experiences"]},
                {"week": 2, "focus": "Future Tense", "goals": ["Make plans", "Discuss goals"]},
                {"week": 3, "focus": "Conditional Sentences", "goals": ["Express possibilities", "Make hypotheticals"]},
                {"week": 4, "focus": "Complex Sentences", "goals": ["Combine ideas", "Use connectors"]},
                {"week": 5, "focus": "Idioms & Slang", "goals": ["Understand idioms", "Use casual language"]},
                {"week": 6, "focus": "Professional Context", "goals": ["Business emails", "Work conversations"]},
                {"week": 7, "focus": "Media Comprehension", "goals": ["Watch videos", "Read articles"]},
                {"week": 8, "focus": "Debate & Discussion", "goals": ["Express opinions", "Argue points"]},
                {"week": 9, "focus": "Cultural Deep Dive", "goals": ["Understand customs", "Learn history"]},
                {"week": 10, "focus": "Creative Writing", "goals": ["Write stories", "Compose emails"]},
                {"week": 11, "focus": "Accent Reduction", "goals": ["Improve pronunciation", "Work on intonation"]},
                {"week": 12, "focus": "Review & Assessment", "goals": ["Test skills", "Set new goals"]}
            ]
        },
        "advanced": {
            "title": f"{language.title()} Advanced Fluency",
            "description": "Achieve native-like proficiency",
            "weeks": 16,
            "daily_minutes": time_per_day,
            "focus_areas": focus_list,
            "total_hours": (16 * 7 * time_per_day) / 60,
            "milestones": [
                {"week": 1, "focus": "Advanced Grammar", "goals": ["Master subjunctive", "Use complex structures"]},
                {"week": 2, "focus": "Academic Language", "goals": ["Read papers", "Write essays"]},
                {"week": 3, "focus": "Literary Analysis", "goals": ["Analyze literature", "Understand poetry"]},
                {"week": 4, "focus": "Professional Negotiation", "goals": ["Business negotiation", "Contract discussion"]},
                {"week": 5, "focus": "Public Speaking", "goals": ["Give presentations", "Speak at events"]},
                {"week": 6, "focus": "Technical Language", "goals": ["Discuss technology", "Explain processes"]},
                {"week": 7, "focus": "Cultural Nuances", "goals": ["Understand humor", "Get sarcasm"]},
                {"week": 8, "focus": "Translation Skills", "goals": ["Translate documents", "Interpret conversations"]},
                {"week": 9, "focus": "Creative Expression", "goals": ["Write poetry", "Create content"]},
                {"week": 10, "focus": "Debate Mastery", "goals": ["Formal debates", "Persuasive speaking"]},
                {"week": 11, "focus": "Media Production", "goals": ["Record podcasts", "Make videos"]},
                {"week": 12, "focus": "Teaching Skills", "goals": ["Explain concepts", "Correct others"]},
                {"week": 13, "focus": "Specialized Vocabulary", "goals": ["Learn jargon", "Master terminology"]},
                {"week": 14, "focus": "Accent Perfection", "goals": ["Sound native", "Master intonation"]},
                {"week": 15, "focus": "Cultural Immersion", "goals": ["Live virtually", "Think in language"]},
                {"week": 16, "focus": "Mastery Assessment", "goals": ["Final evaluation", "Certification prep"]}
            ]
        }
    }
    
    return paths.get(level, paths["beginner"])

@app.post("/progress/update")
async def update_progress(
    update: ProgressUpdate,
    current_user: str = Depends(get_current_user)
):
    """Update user learning progress"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == current_user).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update progress in database
        progress = user.progress
        if "skills" not in progress:
            progress["skills"] = {}
        
        progress["skills"][update.skill] = {
            "level": update.level,
            "score": update.score,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Update XP
        progress["xp"] = progress.get("xp", 0) + update.xp_gained
        
        # Check level up (every 1000 XP)
        new_level = progress["xp"] // 1000 + 1
        if new_level > progress.get("level", 1):
            progress["level"] = new_level
            # Award achievement for level up
            achievements = progress.get("achievements", [])
            achievements.append(f"Reached level {new_level}")
            progress["achievements"] = achievements
        
        user.progress = progress
        db.commit()
        
        # Store in Redis for real-time updates
        if redis_client:
            await redis_client.hset(
                f"user:{current_user}:progress",
                update.skill,
                json.dumps({
                    "level": update.level,
                    "score": update.score,
                    "timestamp": datetime.utcnow().isoformat(),
                    "xp_gained": update.xp_gained
                })
            )
        
        # Calculate overall progress
        skill_scores = [s["score"] for s in progress["skills"].values() if isinstance(s, dict)]
        average_score = sum(skill_scores) / len(skill_scores) if skill_scores else 0
        
        return {
            "success": True,
            "average_score": round(average_score, 1),
            "skill_count": len(progress["skills"]),
            "total_xp": progress["xp"],
            "level": progress["level"],
            "updated_at": datetime.utcnow().isoformat()
        }
    finally:
        db.close()

@app.get("/analytics/dashboard/{user_id}")
async def get_dashboard(
    user_id: str,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Get comprehensive learning analytics"""
    # Authorization check
    if current_user and current_user != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get data from database
        progress = user.progress
        
        # Get recent conversations
        recent_conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(
            Conversation.created_at.desc()
        ).limit(10).all()
        
        # Calculate statistics
        total_sessions = progress.get("total_sessions", 0)
        streak = progress.get("streak", 0)
        total_time = progress.get("total_time", 0)  # in minutes
        
        # Generate insights
        insights = await generate_learning_insights(user_id, db)
        
        # Get weekly activity
        weekly_activity = await get_weekly_activity_db(user_id, db)
        
        return {
            "user_id": user_id,
            "username": user.username,
            "progress_summary": {
                "total_sessions": total_sessions,
                "current_streak": streak,
                "total_time_minutes": total_time,
                "total_time_hours": round(total_time / 60, 1),
                "level": progress.get("level", 1),
                "xp": progress.get("xp", 0),
                "skills_learned": len(progress.get("skills", {})),
                "average_score": calculate_average_score(progress.get("skills", {})),
                "achievements_count": len(progress.get("achievements", []))
            },
            "weekly_activity": weekly_activity,
            "skill_distribution": progress.get("skills", {}),
            "recent_conversations": [
                {
                    "id": conv.id[:8],
                    "mode": conv.mode,
                    "language": conv.language,
                    "duration": conv.duration,
                    "created_at": conv.created_at.isoformat()
                }
                for conv in recent_conversations[:5]
            ],
            "insights": insights,
            "recommendations": await get_recommendations(user_id, progress),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    checks = {
        "api": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0"
    }
    
    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
            checks["redis"] = "connected"
        else:
            checks["redis"] = "not_configured"
    except:
        checks["redis"] = "disconnected"
    
    # Check Database
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        checks["database"] = "connected"
    except:
        checks["database"] = "disconnected"
    
    # Check AI Services
    try:
        if client:
            # Quick test query
            test_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            checks["groq_api"] = "connected"
        else:
            checks["groq_api"] = "not_configured"
    except:
        checks["groq_api"] = "disconnected"
    
    # System info
    checks["system"] = {
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.cpu_percent(),
        "uptime_seconds": time.time() - app_start_time
    }
    
    overall_status = "healthy" if all(
        status in ["healthy", "connected", "not_configured"] 
        for status in checks.values() if isinstance(status, str)
    ) else "degraded"
    
    checks["status"] = overall_status
    
    return checks

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    import psutil
    
    metrics = {
        "lingobot_active_connections": len(manager.active_connections),
        "lingobot_total_sessions": await redis_client.get("global:sessions:total") or 0 if redis_client else 0,
        "lingobot_uptime_seconds": time.time() - app_start_time,
        "lingobot_memory_usage_bytes": psutil.Process().memory_info().rss,
        "lingobot_cpu_percent": psutil.cpu_percent(),
        "lingobot_disk_usage_percent": psutil.disk_usage('/').percent,
    }
    
    # Format for Prometheus
    prometheus_output = "\n".join(
        f"{key} {value}" for key, value in metrics.items()
    )
    
    return Response(content=prometheus_output, media_type="text/plain")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket for real-time features"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "chat":
                # Real-time chat processing
                response = await process_realtime_chat(data["message"])
                await manager.send_personal_message(
                    json.dumps({"type": "chat_response", "data": response}),
                    client_id
                )
                
            elif message_type == "typing":
                # Typing indicator
                await manager.broadcast(
                    json.dumps({"type": "typing", "user_id": client_id})
                )
                
            elif message_type == "audio_stream":
                # Process audio stream
                await process_audio_stream(data["audio_chunk"], client_id)
                
            elif message_type == "authenticate":
                # Link user to session
                token = data.get("token")
                try:
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    user_id = payload.get("sub")
                    manager.user_sessions[user_id] = client_id
                    await websocket.send_json({"type": "auth_success", "user_id": user_id})
                except:
                    await websocket.send_json({"type": "auth_failed"})
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(json.dumps({"type": "user_left", "user_id": client_id}))

# 10. HELPER FUNCTIONS

async def store_conversation_async(session_id: str, user_id: str, request: ChatRequest, response: Dict):
    """Store conversation in database asynchronously"""
    db = SessionLocal()
    try:
        conv = Conversation(
            session_id=session_id,
            user_id=user_id,
            messages=[
                {"role": "user", "content": request.user_text},
                {"role": "assistant", "content": response["ai_reply"]}
            ],
            language=request.language,
            mode=request.mode.value,
            difficulty=request.difficulty.value,
            analytics=response,
            duration=60  # Simulated duration
        )
        db.add(conv)
        db.commit()
        logger.info("Conversation stored", session_id=session_id[:8], user_id=user_id)
    except Exception as e:
        logger.error("Failed to store conversation", error=str(e))
    finally:
        db.close()

async def update_user_progress_async(user_id: str, request: ChatRequest, response: Dict):
    """Update user progress based on interaction asynchronously"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return
        
        progress = user.progress
        
        # Increment session count
        progress["total_sessions"] = progress.get("total_sessions", 0) + 1
        
        # Add time (simulate 5 minutes per session)
        progress["total_time"] = progress.get("total_time", 0) + 5
        
        # Update XP
        xp_gained = calculate_xp(response)
        progress["xp"] = progress.get("xp", 0) + xp_gained
        
        # Update streak
        today = datetime.utcnow().date().isoformat()
        last_active_str = progress.get("last_active_date")
        
        if last_active_str != today:
            if last_active_str and (datetime.fromisoformat(last_active_str).date() == 
                                   datetime.utcnow().date() - timedelta(days=1)):
                progress["streak"] = progress.get("streak", 0) + 1
            else:
                progress["streak"] = 1
            
            progress["last_active_date"] = today
        
        # Update skills based on analytics
        grammar_score = response.get("grammar_feedback", {}).get("score", 0)
        if grammar_score > 0:
            skills = progress.get("skills", {})
            grammar_data = skills.get("grammar", {"score": 0, "level": 1})
            grammar_data["score"] = min(100, grammar_data["score"] + (grammar_score / 10))
            
            # Level up skill if score > 90
            if grammar_data["score"] >= 90 and grammar_data["level"] < 5:
                grammar_data["level"] += 1
                grammar_data["score"] = 0  # Reset for next level
            
            skills["grammar"] = grammar_data
            progress["skills"] = skills
        
        user.progress = progress
        db.commit()
        
        # Update Redis
        if redis_client:
            await redis_client.hincrby(f"user:{user_id}:stats", "session_count", 1)
            await redis_client.hset(f"user:{user_id}:stats", "last_active", today)
            
            # Update streak in Redis
            current_streak = int(await redis_client.hget(f"user:{user_id}:stats", "streak") or 0)
            await redis_client.hset(f"user:{user_id}:stats", "streak", progress["streak"])
        
        logger.info("User progress updated", user_id=user_id, xp_gained=xp_gained)
        
    except Exception as e:
        logger.error("Failed to update user progress", error=str(e))
    finally:
        db.close()

def extract_learning_points(ai_reply: str, user_input: str) -> List[Dict]:
    """Extract specific learning points from conversation"""
    points = []
    
    import re
    
    # Vocabulary extraction
    vocabulary_patterns = [
        r'\b(\w+)\b.*?(?:means|refers to|is called)',
        r'Key (?:word|phrase):?\s*(\w+(?:\s+\w+)*)',
        r'Remember.*?\b(\w+(?:\s+\w+)*)\b',
        r'Word.*?\b(\w+)\b.*?(?:means|is)'
    ]
    
    for pattern in vocabulary_patterns:
        matches = re.findall(pattern, ai_reply, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            
            # Clean up the match
            match = match.strip()
            if len(match.split()) <= 3 and len(match) > 2:  # Single words or short phrases
                points.append({
                    "type": "vocabulary",
                    "content": match,
                    "context": "From conversation",
                    "example": find_example_in_reply(ai_reply, match)
                })
    
    # Grammar points
    grammar_keywords = ["grammar", "tense", "plural", "singular", "conjugation", 
                       "preposition", "article", "pronoun", "adjective", "adverb"]
    
    for keyword in grammar_keywords:
        if keyword in ai_reply.lower():
            points.append({
                "type": "grammar",
                "topic": keyword.title(),
                "explanation": extract_paragraph_containing(ai_reply, keyword),
                "examples": find_examples_in_reply(ai_reply, keyword)
            })
    
    # Cultural points
    cultural_keywords = ["culture", "custom", "tradition", "norm", "etiquette"]
    for keyword in cultural_keywords:
        if keyword in ai_reply.lower():
            points.append({
                "type": "cultural",
                "topic": extract_sentence_containing(ai_reply, keyword),
                "insight": extract_paragraph_containing(ai_reply, keyword)
            })
    
    return points[:5]  # Limit to 5 points

def suggest_topics(user_input: str) -> List[str]:
    """Suggest next conversation topics based on current input"""
    topics_db = {
        "food": ["restaurant ordering", "cooking vocabulary", "dietary restrictions", "food culture"],
        "travel": ["directions", "accommodation", "transportation", "sightseeing"],
        "work": ["job interviews", "office communication", "professional email", "business meetings"],
        "daily": ["morning routine", "shopping", "social plans", "family life"],
        "hobbies": ["sports", "music", "reading", "movies", "gaming"],
        "education": ["school life", "university", "learning methods", "study tips"],
        "technology": ["social media", "smartphones", "computers", "internet"],
        "health": ["doctor visits", "exercise", "nutrition", "mental health"]
    }
    
    input_lower = user_input.lower()
    suggested = []
    
    for category, category_topics in topics_db.items():
        if any(word in input_lower for word in category.split()):
            suggested.extend(category_topics[:2])
    
    # Default suggestions if no match
    if not suggested:
        suggested = ["introductions", "weather", "future plans", "personal interests"]
    
    return suggested[:4]

async def get_cultural_notes(text: str, target_lang: str) -> List[str]:
    """Get cultural context for translation"""
    notes = []
    
    # Simple cultural database (in production, this would be a proper database)
    cultural_data = {
        "en": {
            "thanks": "In English-speaking cultures, 'thank you' is used very frequently in daily interactions.",
            "please": "Using 'please' makes requests much more polite and is expected in most situations.",
            "sorry": "English speakers apologize often, even for minor inconveniences or when not at fault.",
            "how_are_you": "This is often a greeting rather than a genuine inquiry about health.",
            "time": "Punctuality is highly valued in English-speaking business cultures."
        },
        "es": {
            "gracias": "In Spanish culture, expressing gratitude is important and frequent.",
            "por_favor": "Always use 'por favor' when making requests to be polite.",
            "perdón": "Used similarly to English 'sorry' but can also mean 'excuse me'.",
            "tú_vs_usted": "Use 'tú' for informal situations and 'usted' for formal/respected persons.",
            "siesta": "Afternoon rest is common in many Spanish-speaking countries."
        },
        "fr": {
            "bonjour": "Always greet with 'bonjour' before starting any conversation.",
            "merci": "Say 'merci' frequently as gratitude is important in French culture.",
            "s_vous_plaît": "The polite form for 'please' is essential in all requests.",
            "tu_vs_vous": "'Tu' is informal, 'vous' is formal or plural - use carefully.",
            "food": "Food and dining are central to French culture and conversation."
        }
    }
    
    # Check for keywords in text
    text_lower = text.lower()
    if target_lang in cultural_data:
        for keyword, note in cultural_data[target_lang].items():
            if keyword.replace("_", " ") in text_lower:
                notes.append(note)
    
    # Add general cultural tips
    if target_lang in ["ja", "ko", "zh"]:
        notes.append("In many Asian cultures, indirect communication and saving face are important concepts.")
    
    if target_lang in ["de", "nl", "se"]:
        notes.append("Direct communication is often appreciated in these cultures.")
    
    return notes[:3] if notes else ["No specific cultural notes for this text."]

async def get_translation_alternatives(text: str, target_lang: str) -> List[Dict]:
    """Get alternative translations"""
    alternatives = []
    
    # Simple synonyms and alternatives (in production, use proper translation API)
    common_alternatives = {
        "hello": ["hi", "hey", "greetings", "good day"],
        "thank you": ["thanks", "much appreciated", "many thanks"],
        "goodbye": ["bye", "see you", "farewell", "take care"],
        "please": ["if you would", "kindly", "if you don't mind"],
        "sorry": ["apologies", "my apologies", "pardon me", "excuse me"]
    }
    
    text_lower = text.lower().strip(".?!")
    
    if text_lower in common_alternatives:
        for alt in common_alternatives[text_lower]:
            alternatives.append({
                "translation": alt.capitalize(),
                "formality": "informal" if alt in ["hi", "hey", "bye", "thanks"] else "neutral",
                "context": "Casual conversation"
            })
    
    # Add formal alternatives
    if text_lower == "hello":
        alternatives.append({
            "translation": "Good morning/afternoon/evening",
            "formality": "formal",
            "context": "Professional settings"
        })
    
    return alternatives[:3]

def calculate_xp(response: Dict) -> int:
    """Calculate XP gained from interaction"""
    base_xp = 10
    
    # Grammar bonus
    grammar_score = response.get("grammar_feedback", {}).get("score", 0)
    grammar_bonus = grammar_score / 10
    
    # Sentiment bonus
    sentiment = response.get("sentiment_feedback", {}).get("sentiment", "neutral")
    sentiment_bonus = 5 if sentiment == "positive" else 0
    
    # Length bonus
    ai_reply = response.get("ai_reply", "")
    length_bonus = min(len(ai_reply) // 50, 10)
    
    # Vocabulary bonus
    vocab_count = len(response.get("vocabulary_highlight", []))
    vocab_bonus = vocab_count * 2
    
    # Mode bonus
    mode = response.get("mode", "tutor")
    mode_bonus = 3 if mode in ["examiner", "debate"] else 0
    
    total_xp = int(base_xp + grammar_bonus + sentiment_bonus + length_bonus + vocab_bonus + mode_bonus)
    
    return min(total_xp, 50)  # Cap at 50 XP per interaction

async def generate_learning_insights(user_id: str, db: Session) -> List[str]:
    """Generate personalized learning insights"""
    insights = []
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return ["Welcome! Start by practicing greetings and introductions."]
    
    progress = user.progress
    skills = progress.get("skills", {})
    
    # Skill-based insights
    if not skills:
        insights.append("You're just starting! Try different conversation topics to build vocabulary.")
    else:
        # Check individual skills
        for skill_name, skill_data in skills.items():
            if isinstance(skill_data, dict):
                score = skill_data.get("score", 0)
                if score < 30:
                    insights.append(f"Your {skill_name} needs more practice. Try focusing on it.")
                elif score > 80:
                    insights.append(f"Great job on {skill_name}! You're doing well.")
        
        avg_score = calculate_average_score(skills)
        if avg_score < 50:
            insights.append("Consider practicing more frequently to improve your skills.")
        elif avg_score > 75:
            insights.append("Excellent progress! Try more challenging topics.")
    
    # Time-based insights
    total_time = progress.get("total_time", 0)
    if total_time < 60:  # Less than 1 hour
        insights.append("Keep going! Consistency is key to language learning.")
    elif total_time > 300:  # More than 5 hours
        insights.append("You've put in significant practice time! Your dedication is paying off.")
    
    # Streak insights
    streak = progress.get("streak", 0)
    if streak == 0:
        insights.append("Start a learning streak by practicing every day!")
    elif streak == 1:
        insights.append("You practiced today! Try to make it a daily habit.")
    elif streak >= 7:
        insights.append(f"Amazing {streak}-day streak! Your consistency is impressive.")
    
    # Session frequency
    total_sessions = progress.get("total_sessions", 0)
    if total_sessions > 0:
        avg_session_gap = total_time / total_sessions if total_time > 0 else 0
        if avg_session_gap > 10:  # Long sessions
            insights.append("You prefer longer practice sessions. This helps with immersion!")
        else:  # Short frequent sessions
            insights.append("Frequent short practice sessions are great for retention.")
    
    return insights[:3]  # Return top 3 insights

def calculate_average_score(skills: Dict) -> float:
    """Calculate average score from skills data"""
    if not skills:
        return 0.0
    
    scores = []
    for skill_data in skills.values():
        if isinstance(skill_data, dict):
            score = skill_data.get("score", 0)
            scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0

async def get_weekly_activity_db(user_id: str, db: Session) -> Dict:
    """Get weekly activity data from database"""
    # Simplified version - in production, query the database
    return {
        "monday": 3,
        "tuesday": 5,
        "wednesday": 2,
        "thursday": 4,
        "friday": 6,
        "saturday": 1,
        "sunday": 0
    }

async def get_recommendations(user_id: str, progress: Dict) -> List[str]:
    """Get personalized learning recommendations"""
    recommendations = []
    
    skills = progress.get("skills", {})
    
    # Skill-based recommendations
    for skill_name, skill_data in skills.items():
        if isinstance(skill_data, dict):
            score = skill_data.get("score", 0)
            if score < 50:
                recommendations.append(f"Practice {skill_name} exercises")
    
    # General recommendations
    level = progress.get("level", 1)
    
    if level <= 2:
        recommendations.extend([
            "Practice basic greetings and introductions",
            "Learn 10 new words daily",
            "Try the 'Tutor' mode for guided practice"
        ])
    elif level <= 4:
        recommendations.extend([
            "Try the 'Roleplay' mode for real-life scenarios",
            "Practice past tense conversations",
            "Listen to pronunciation carefully and repeat"
        ])
    else:
        recommendations.extend([
            "Challenge yourself with 'Debate' mode",
            "Try translating short articles",
            "Practice with native speakers if possible",
            "Review and consolidate what you've learned"
        ])
    
    # Add based on total time
    total_time = progress.get("total_time", 0)
    if total_time > 120:  # More than 2 hours
        recommendations.append("Consider setting specific learning goals")
    
    return recommendations[:4]

async def process_realtime_chat(message: str) -> Dict:
    """Process chat message in real-time"""
    if not client:
        return {
            "text": "AI service is currently unavailable. Please try again later.",
            "is_typing": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Quick response without full processing
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": message}],
            temperature=0.7,
            max_tokens=100
        )
        
        return {
            "text": response.choices[0].message.content,
            "is_typing": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Realtime chat error", error=str(e))
        return {
            "text": "I'm having trouble responding right now. Please try again.",
            "is_typing": False,
            "timestamp": datetime.utcnow().isoformat()
        }

async def process_audio_stream(audio_chunk: str, client_id: str):
    """Process streaming audio"""
    # Placeholder for real-time audio processing
    # In production, this would buffer audio and process in chunks
    pass

def extract_paragraph_containing(text: str, keyword: str) -> str:
    """Extract paragraph containing a keyword"""
    paragraphs = text.split('\n\n')
    for para in paragraphs:
        if keyword.lower() in para.lower():
            return para.strip()
    return ""

def find_example_in_reply(text: str, word: str) -> str:
    """Find example usage of a word in text"""
    sentences = text.split('. ')
    for sentence in sentences:
        if word.lower() in sentence.lower():
            return sentence.strip()
    return ""

def find_examples_in_reply(text: str, keyword: str) -> List[str]:
    """Find multiple examples in text"""
    examples = []
    sentences = text.split('. ')
    
    for sentence in sentences:
        if keyword.lower() in sentence.lower():
            examples.append(sentence.strip())
    
    return examples[:2]  # Return up to 2 examples

def extract_sentence_containing(text: str, keyword: str) -> str:
    """Extract sentence containing keyword"""
    sentences = text.split('. ')
    for sentence in sentences:
        if keyword.lower() in sentence.lower():
            return sentence.strip()
    return ""

def assess_translation_difficulty(text: str) -> str:
    """Assess difficulty level of translation"""
    word_count = len(text.split())
    
    if word_count <= 5:
        return "easy"
    elif word_count <= 15:
        return "medium"
    elif word_count <= 30:
        return "hard"
    else:
        return "expert"

# 11. STATIC FILES AND ROUTES
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard"""
    try:
        with open("dashboard/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except:
        return HTMLResponse(content="<h1>LingoBot Pro API</h1><p>API is running. Visit /api/docs for documentation.</p>", status_code=200)

@app.get("/dashboard")
async def get_dashboard_page():
    """Redirect to main dashboard"""
    return RedirectResponse(url="/")

# 12. STARTUP AND SHUTDOWN
app_start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting LingoBot Pro API...")
    
    # Test Redis connection
    if redis_client:
        try:
            await redis_client.ping()
            logger.info("Redis connection successful")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    else:
        logger.warning("Redis not configured")
    
    # Load ML models in background
    asyncio.create_task(ai_service.load_models())
    
    # Initialize global counters in Redis
    if redis_client:
        try:
            await redis_client.setnx("global:sessions:total", 0)
            await redis_client.setnx("global:chats:total", 0)
        except:
            pass
    
    logger.info("LingoBot Pro API ready!", version="3.0.0")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down LingoBot Pro API...")
    
    if redis_client:
        await redis_client.close()
    
    logger.info("Cleanup complete")

# 13. CELERY CONFIGURATION (for background tasks)
from celery import Celery

celery_app = Celery(
    "lingobot",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["server"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
)

@celery_app.task(name="process_audio_background")
def process_audio_background(audio_data: str, language: str):
    """Background task for audio processing"""
    # Your audio processing logic here
    pass

@celery_app.task(name="generate_tts_background")
def generate_tts_background(text: str, language: str):
    """Background task for TTS generation"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode('utf-8')
    except Exception as e:
        logger.error("Background TTS error", error=str(e))
        return None

@celery_app.task(name="analyze_conversation_batch")
def analyze_conversation_batch(conversation_ids: List[str]):
    """Batch analysis of conversations"""
    # Process multiple conversations for analytics
    pass

# 14. MAIN ENTRY POINT
if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info",
        "access_log": True
    }
    
    logger.info(f"Starting server on {config['host']}:{config['port']}")
    uvicorn.run("server:app", **config)
