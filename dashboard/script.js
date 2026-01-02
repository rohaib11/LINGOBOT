// dashboard/script.js
class LingoBotApp {
   constructor() {
        // CORRECTED: Removed ':8000'. 
        // Requests now go to http://your-site.com/api, which Nginx handles.
        this.apiBaseUrl = window.location.protocol + '//' + window.location.host + '/api';
        
        this.socket = null;
        this.sessionId = this.generateSessionId();
        this.currentMode = 'tutor';
        this.currentLanguage = 'en';
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.authToken = localStorage.getItem('lingobot_token');
        this.username = localStorage.getItem('lingobot_username');
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadUserData();
        this.connectWebSocket();
        this.updateUI();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    bindEvents() {
        // Login/Register
        document.getElementById('loginBtn').addEventListener('click', () => this.showLoginModal());
        document.querySelector('.close-modal').addEventListener('click', () => this.hideLoginModal());
        document.getElementById('loginForm').addEventListener('submit', (e) => this.handleLogin(e));
        document.getElementById('registerForm').addEventListener('submit', (e) => this.handleRegister(e));
        
        // Chat mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.setChatMode(e));
        });
        
        // Language selector
        document.getElementById('languageSelect').addEventListener('change', (e) => {
            this.currentLanguage = e.target.value;
            this.updateChatInfo();
        });
        
        // Chat input
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        document.getElementById('sendBtn').addEventListener('click', () => this.sendMessage());
        document.getElementById('recordBtn').addEventListener('click', () => this.toggleRecording());
        document.getElementById('stopRecordingBtn').addEventListener('click', () => this.stopRecording());
        document.getElementById('translateBtn').addEventListener('click', () => this.showTranslation());
        document.getElementById('repeatBtn').addEventListener('click', () => this.repeatAudio());
        document.getElementById('refreshStats').addEventListener('click', () => this.loadStats());
        
        // Modal tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e));
        });
        
        // Close modal on outside click
        document.getElementById('loginModal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) {
                this.hideLoginModal();
            }
        });
    }

 connectWebSocket() {
        try {
            // FIX: Dynamic protocol handling (ws vs wss)
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            
            // FIX: Remove port 8000. Nginx (on port 80/443) will route /ws/ to the backend.
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
            
            console.log('Connecting to WebSocket:', wsUrl); // Debugging
            
            this.socket = new WebSocket(wsUrl);
            
            this.socket.onopen = () => {
                console.log('WebSocket connected');
                this.addSystemMessage('Connected to LingoBot server');
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.socket.onclose = () => {
                console.log('WebSocket disconnected');
                this.addSystemMessage('Disconnected from server. Reconnecting...');
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'chat_response':
                this.handleChatResponse(data.data);
                break;
            case 'typing':
                this.showTypingIndicator(data.user_id);
                break;
            case 'user_left':
                this.addSystemMessage(`User ${data.user_id} left`);
                break;
        }
    }

    async sendMessage() {
        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        input.value = '';
        
        // Show typing indicator
        this.showTypingIndicator('ai');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat/advanced`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.authToken && { 'Authorization': `Bearer ${this.authToken}` })
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    user_text: message,
                    language: this.currentLanguage,
                    mode: this.currentMode,
                    context: {
                        previous_mode: this.currentMode,
                        difficulty: 'intermediate'
                    }
                })
            });
            
            const data = await response.json();
            this.handleChatResponse(data);
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.addSystemMessage('Error: Could not connect to server');
        }
    }

    async sendAudioMessage(audioBlob) {
        const audioBase64 = await this.blobToBase64(audioBlob);
        
        // Show processing message
        this.addSystemMessage('Processing audio...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/audio/transcribe`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.authToken && { 'Authorization': `Bearer ${this.authToken}` })
                },
                body: JSON.stringify({
                    audio_base64: audioBase64.split(',')[1],
                    language: this.currentLanguage
                })
            });
            
            const data = await response.json();
            
            // Add transcribed text as user message
            this.addMessage(data.text, 'user');
            
            // Now send it through chat
            this.sendTextAsMessage(data.text);
            
        } catch (error) {
            console.error('Error sending audio:', error);
            this.addSystemMessage('Error: Could not process audio');
        }
    }

    async sendTextAsMessage(text) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat/advanced`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.authToken && { 'Authorization': `Bearer ${this.authToken}` })
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    user_text: text,
                    language: this.currentLanguage,
                    mode: this.currentMode
                })
            });
            
            const data = await response.json();
            this.handleChatResponse(data);
            
        } catch (error) {
            console.error('Error sending message:', error);
        }
    }

    handleChatResponse(data) {
        // Remove typing indicator
        this.hideTypingIndicator();
        
        // Add AI response
        this.addMessage(data.ai_reply, 'ai');
        
        // Play audio if available
        if (data.audio_base64) {
            this.playAudio(data.audio_base64);
        }
        
        // Update stats from response
        if (data.enhanced_context) {
            this.updateStatsFromResponse(data.enhanced_context);
        }
        
        // Save to conversation history
        this.saveToHistory(data);
    }

    addMessage(text, type) {
        const messagesDiv = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="avatar">
                <i class="fas fa-${type === 'ai' ? 'robot' : 'user'}"></i>
            </div>
            <div class="content">
                <p>${this.formatMessage(text)}</p>
                <span class="time">${time}</span>
            </div>
        `;
        
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    addSystemMessage(text) {
        const messagesDiv = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.innerHTML = `<div class="content"><p><i>${text}</i></p></div>`;
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    formatMessage(text) {
        // Convert markdown-like formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    showTypingIndicator(userId) {
        const messagesDiv = document.getElementById('chatMessages');
        let typingDiv = messagesDiv.querySelector('.typing-indicator');
        
        if (!typingDiv) {
            typingDiv = document.createElement('div');
            typingDiv.className = 'message ai typing-indicator';
            typingDiv.innerHTML = `
                <div class="avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="content">
                    <p><i>${userId === 'ai' ? 'AI is typing...' : 'User is typing...'}</i></p>
                </div>
            `;
            messagesDiv.appendChild(typingDiv);
        }
        
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    hideTypingIndicator() {
        const typingDiv = document.querySelector('.typing-indicator');
        if (typingDiv) {
            typingDiv.remove();
        }
    }

    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            this.stopRecording();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                await this.sendAudioMessage(audioBlob);
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateRecordingUI(true);
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.addSystemMessage('Error: Could not access microphone. Please check permissions.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateRecordingUI(false);
        }
    }

    updateRecordingUI(isRecording) {
        const recordBtn = document.getElementById('recordBtn');
        const recordingSection = document.getElementById('recordingSection');
        
        if (isRecording) {
            recordBtn.innerHTML = '<i class="fas fa-microphone-slash"></i> Stop';
            recordBtn.classList.add('btn-danger');
            recordingSection.style.display = 'block';
        } else {
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Record';
            recordBtn.classList.remove('btn-danger');
            recordingSection.style.display = 'none';
        }
    }

    playAudio(base64Audio) {
        const audioPlayer = document.getElementById('responseAudio');
        const audioPlayerDiv = document.getElementById('audioPlayer');
        
        audioPlayer.src = `data:audio/mp3;base64,${base64Audio}`;
        audioPlayerDiv.style.display = 'flex';
        
        audioPlayer.play().catch(error => {
            console.error('Error playing audio:', error);
        });
    }

    repeatAudio() {
        const audioPlayer = document.getElementById('responseAudio');
        audioPlayer.currentTime = 0;
        audioPlayer.play();
    }

    setChatMode(event) {
        const mode = event.currentTarget.dataset.mode;
        
        // Update active button
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.currentTarget.classList.add('active');
        
        this.currentMode = mode;
        this.updateChatInfo();
        
        // Send mode change notification
        this.addSystemMessage(`Switched to ${mode} mode`);
    }

    updateChatInfo() {
        document.getElementById('currentMode').textContent = 
            this.currentMode.charAt(0).toUpperCase() + this.currentMode.slice(1) + ' Mode';
        
        const languageNames = {
            en: 'English', es: 'Spanish', fr: 'French', de: 'German',
            it: 'Italian', pt: 'Portuguese', ru: 'Russian',
            zh: 'Chinese', ja: 'Japanese', ko: 'Korean'
        };
        
        document.getElementById('currentLanguage').textContent = 
            languageNames[this.currentLanguage] || this.currentLanguage;
    }

    async showTranslation() {
        const input = document.getElementById('messageInput');
        const text = input.value.trim();
        
        if (!text) {
            alert('Please enter text to translate');
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/translate/advanced`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    source_lang: 'auto',
                    target_lang: this.currentLanguage
                })
            });
            
            const data = await response.json();
            
            // Show translation in a system message
            this.addSystemMessage(`Translation: ${data.translated}`);
            
        } catch (error) {
            console.error('Error translating:', error);
            this.addSystemMessage('Error: Could not translate text');
        }
    }

    async handleLogin(event) {
        event.preventDefault();
        
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.authToken = data.access_token;
                this.username = data.username;
                
                localStorage.setItem('lingobot_token', this.authToken);
                localStorage.setItem('lingobot_username', this.username);
                
                this.hideLoginModal();
                this.updateUI();
                this.addSystemMessage(`Welcome back, ${this.username}!`);
                
            } else {
                const error = await response.json();
                alert(`Login failed: ${error.detail}`);
            }
            
        } catch (error) {
            console.error('Login error:', error);
            alert('Login failed. Please try again.');
        }
    }

    async handleRegister(event) {
        event.preventDefault();
        
        const username = document.getElementById('registerUsername').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        
        if (password !== confirmPassword) {
            alert('Passwords do not match');
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, username, password })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.authToken = data.access_token;
                this.username = data.username;
                
                localStorage.setItem('lingobot_token', this.authToken);
                localStorage.setItem('lingobot_username', this.username);
                
                this.hideLoginModal();
                this.updateUI();
                this.addSystemMessage(`Welcome to LingoBot, ${this.username}!`);
                
            } else {
                const error = await response.json();
                alert(`Registration failed: ${error.detail}`);
            }
            
        } catch (error) {
            console.error('Registration error:', error);
            alert('Registration failed. Please try again.');
        }
    }

    showLoginModal() {
        document.getElementById('loginModal').classList.add('active');
    }

    hideLoginModal() {
        document.getElementById('loginModal').classList.remove('active');
    }

    switchTab(event) {
        const tab = event.currentTarget.dataset.tab;
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        event.currentTarget.classList.add('active');
        
        // Show active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tab}Form`).classList.add('active');
    }

    loadUserData() {
        if (this.username) {
            document.getElementById('username').textContent = this.username;
            document.getElementById('loginBtn').textContent = 'Logout';
            document.getElementById('loginBtn').onclick = () => this.logout();
        }
    }

    logout() {
        localStorage.removeItem('lingobot_token');
        localStorage.removeItem('lingobot_username');
        this.authToken = null;
        this.username = null;
        
        this.updateUI();
        this.addSystemMessage('Logged out successfully');
    }

    updateUI() {
        const usernameSpan = document.getElementById('username');
        const loginBtn = document.getElementById('loginBtn');
        
        if (this.username) {
            usernameSpan.textContent = this.username;
            loginBtn.innerHTML = '<i class="fas fa-sign-out-alt"></i> Logout';
            loginBtn.onclick = () => this.logout();
        } else {
            usernameSpan.textContent = 'Guest';
            loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
            loginBtn.onclick = () => this.showLoginModal();
        }
    }

    async loadStats() {
        if (!this.authToken) return;
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/analytics/dashboard/${this.sessionId}`, {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateStatsUI(data);
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    updateStatsUI(data) {
        // Update basic stats
        if (data.progress_summary) {
            document.getElementById('streakValue').textContent = 
                `${data.progress_summary.current_streak} days`;
            document.getElementById('xpValue').textContent = 
                data.progress_summary.xp || 0;
            document.getElementById('sessionsValue').textContent = 
                data.progress_summary.total_sessions;
            document.getElementById('timeValue').textContent = 
                `${data.progress_summary.total_minutes || 0} min`;
        }
        
        // Update skill progress
        if (data.skill_distribution) {
            const skills = data.skill_distribution;
            
            document.getElementById('grammarProgress').style.width = 
                `${skills.grammar?.score || 25}%`;
            document.querySelector('#grammarProgress').parentElement.nextElementSibling.textContent = 
                `${skills.grammar?.score || 25}%`;
            
            document.getElementById('vocabProgress').style.width = 
                `${skills.vocabulary?.score || 40}%`;
            document.querySelector('#vocabProgress').parentElement.nextElementSibling.textContent = 
                `${skills.vocabulary?.score || 40}%`;
            
            document.getElementById('pronunciationProgress').style.width = 
                `${skills.pronunciation?.score || 15}%`;
            document.querySelector('#pronunciationProgress').parentElement.nextElementSibling.textContent = 
                `${skills.pronunciation?.score || 15}%`;
            
            document.getElementById('fluencyProgress').style.width = 
                `${skills.fluency?.score || 30}%`;
            document.querySelector('#fluencyProgress').parentElement.nextElementSibling.textContent = 
                `${skills.fluency?.score || 30}%`;
        }
        
        // Update tips
        if (data.insights) {
            const tipsList = document.getElementById('tipsList');
            tipsList.innerHTML = '';
            
            data.insights.slice(0, 4).forEach(insight => {
                const li = document.createElement('li');
                li.textContent = insight;
                tipsList.appendChild(li);
            });
        }
    }

    updateStatsFromResponse(context) {
        // Simulate progress update based on response
        const grammarScore = context.grammar_analysis?.score || 0;
        const sentiment = context.sentiment?.sentiment || 'neutral';
        
        // Update XP
        let currentXP = parseInt(document.getElementById('xpValue').textContent) || 0;
        let xpGain = 10;
        
        if (grammarScore > 80) xpGain += 5;
        if (sentiment === 'positive') xpGain += 3;
        
        document.getElementById('xpValue').textContent = currentXP + xpGain;
        
        // Update session count
        let sessions = parseInt(document.getElementById('sessionsValue').textContent) || 0;
        document.getElementById('sessionsValue').textContent = sessions + 1;
        
        // Update time (simulate 5 minutes per interaction)
        let time = parseInt(document.getElementById('timeValue').textContent) || 0;
        document.getElementById('timeValue').textContent = time + 5 + ' min';
        
        // Update streak if it's a new day (simplified)
        if (Math.random() > 0.7) { // 30% chance to increment streak
            let streak = parseInt(document.getElementById('streakValue').textContent) || 0;
            document.getElementById('streakValue').textContent = streak + 1 + ' days';
        }
    }

    saveToHistory(data) {
        const history = JSON.parse(localStorage.getItem('lingobot_history') || '[]');
        history.push({
            timestamp: new Date().toISOString(),
            mode: this.currentMode,
            language: this.currentLanguage,
            user_text: data.user_text || 'audio',
            ai_reply: data.ai_reply,
            analytics: data.enhanced_context
        });
        
        // Keep only last 100 conversations
        if (history.length > 100) {
            history.shift();
        }
        
        localStorage.setItem('lingobot_history', JSON.stringify(history));
    }

    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.lingobot = new LingoBotApp();
});