/* KeyGen.ai - Client Side Script */
const chatContainer = document.getElementById('chat-container');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const voiceBtn = document.getElementById('voiceBtn');
const settingsOverlay = document.getElementById('settingsOverlay');

let isGenerating = false;
let abortController = null;
let useSpeechSynthesis = false;
let activeUtterance = null;
let recognition = null;
let isListening = false;

// Audio context for micro-sounds (pure JS synthesised, keeping it offline and zero-dependency!)
function playSound(type) {
    if (!document.getElementById('soundEffects').checked) return;
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        
        if (type === 'send') {
            osc.frequency.setValueAtTime(600, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.1);
            gain.gain.setValueAtTime(0.05, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.1);
            osc.start();
            osc.stop(ctx.currentTime + 0.1);
        } else if (type === 'receive') {
            osc.frequency.setValueAtTime(800, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(500, ctx.currentTime + 0.15);
            gain.gain.setValueAtTime(0.05, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.15);
            osc.start();
            osc.stop(ctx.currentTime + 0.15);
        } else if (type === 'click') {
            osc.frequency.setValueAtTime(1000, ctx.currentTime);
            gain.gain.setValueAtTime(0.02, ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.05);
            osc.start();
            osc.stop(ctx.currentTime + 0.05);
        }
    } catch (e) {
        console.warn("Sound play failed", e);
    }
}

// Format date nicely
function getFormattedTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Generate code formatting helper
function formatMarkdown(text) {
    // Basic code block formatting
    const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
    let formattedText = text;
    
    formattedText = formattedText.replace(codeBlockRegex, (match, lang, code) => {
        const safeCode = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const language = lang || 'code';
        const uniqueId = 'code-' + Math.random().toString(36).substr(2, 9);
        return `
            <pre>
                <div class="code-header">
                    <span>${language.toUpperCase()}</span>
                    <button class="message-action-btn" onclick="copyCode('${uniqueId}')">📋 Copy</button>
                </div>
                <code id="${uniqueId}">${safeCode.trim()}</code>
            </pre>
        `;
    });

    // Inline code formatting
    formattedText = formattedText.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold formatting
    formattedText = formattedText.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    return formattedText;
}

// Add message to screen
function addMessage(text, isUser) {
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${isUser ? 'user' : 'ai'}`;
    
    const card = document.createElement('div');
    card.className = 'message-card';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    if (isUser) {
        bubble.textContent = text;
    } else {
        bubble.innerHTML = formatMarkdown(text);
    }
    
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    
    const timeSpan = document.createElement('span');
    timeSpan.textContent = getFormattedTime();
    meta.appendChild(timeSpan);
    
    if (!isUser) {
        // Add Copy response action
        const copyBtn = document.createElement('button');
        copyBtn.className = 'message-action-btn';
        copyBtn.innerHTML = '📋 Copy';
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(text);
            copyBtn.innerHTML = '✓ Copied';
            setTimeout(() => copyBtn.innerHTML = '📋 Copy', 2000);
        };
        meta.appendChild(copyBtn);

        // Add TTS Speak action
        const speakBtn = document.createElement('button');
        speakBtn.className = 'message-action-btn';
        speakBtn.innerHTML = '🔊 Speak';
        speakBtn.onclick = () => speakText(text);
        meta.appendChild(speakBtn);
    }
    
    card.appendChild(bubble);
    card.appendChild(meta);
    wrapper.appendChild(card);
    chatContainer.appendChild(wrapper);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return bubble;
}

// Show/remove typing indicator
function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message-wrapper ai';
    indicator.id = 'typing-indicator-wrapper';
    indicator.innerHTML = `
        <div class="message-card">
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    chatContainer.appendChild(indicator);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator-wrapper');
    if (indicator) indicator.remove();
}

// Toggle states
function setGeneratingState(generating) {
    isGenerating = generating;
    if (generating) {
        sendBtn.classList.add('hidden');
        stopBtn.classList.add('active');
        input.disabled = true;
    } else {
        sendBtn.classList.remove('hidden');
        stopBtn.classList.remove('active');
        input.disabled = false;
        input.focus();
    }
}

// Stop current response generation
function stopGeneration() {
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    stopSpeaking();
    removeTypingIndicator();
    setGeneratingState(false);
    playSound('click');
}

// Stream response using typewriter effect
async function typewriterEffect(element, text, speed = 8) {
    element.innerHTML = '';
    const words = text.split(' ');
    let currentHtml = '';
    
    // Quick preprocessing to handle markdown formatting during typing
    // To make it smoother, we type in chunks of words
    for (let i = 0; i < words.length; i++) {
        if (!isGenerating) break;
        currentHtml += (i === 0 ? '' : ' ') + words[i];
        element.innerHTML = formatMarkdown(currentHtml);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        await new Promise(resolve => setTimeout(resolve, speed));
    }
    // Final print to make sure everything matches markdown format perfectly
    element.innerHTML = formatMarkdown(text);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Send user query to backend
async function sendMessage() {
    const text = input.value.trim();
    if (!text || isGenerating) return;
    
    playSound('send');
    addMessage(text, true);
    input.value = '';
    
    showTypingIndicator();
    setGeneratingState(true);
    
    abortController = new AbortController();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text}),
            signal: abortController.signal
        });
        
        const data = await response.json();
        removeTypingIndicator();
        playSound('receive');
        
        if (isGenerating) {
            const aiBubble = addMessage('', false);
            await typewriterEffect(aiBubble, data.response, 10);
            
            // Speak if TTS is toggled on in settings
            if (useSpeechSynthesis) {
                speakText(data.response);
            }
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            removeTypingIndicator();
            addMessage('⚠️ Connection error. Please verify the local server is running.', false);
        }
    }
    
    setGeneratingState(false);
    abortController = null;
}

// Voice Text-to-Speech implementation
function speakText(text) {
    stopSpeaking();
    // Remove HTML tags for clean reading
    const cleanText = text.replace(/<[^>]*>/g, '').replace(/\*+/g, '');
    activeUtterance = new SpeechSynthesisUtterance(cleanText);
    
    // Fetch settings voice volume/speed if desired
    activeUtterance.rate = 1.0;
    activeUtterance.volume = 1.0;
    
    window.speechSynthesis.speak(activeUtterance);
}

function stopSpeaking() {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
}

// Voice Speech-to-Text input
function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        voiceBtn.style.display = 'none';
        return;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        isListening = true;
        voiceBtn.classList.add('listening');
        voiceBtn.innerHTML = '🎙️';
        input.placeholder = "Listening...";
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        input.value = transcript;
    };
    
    recognition.onerror = (event) => {
        console.error("STT error", event);
        stopListening();
    };
    
    recognition.onend = () => {
        stopListening();
    };
}

function toggleListening() {
    playSound('click');
    if (!recognition) return;
    
    if (isListening) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

function stopListening() {
    isListening = false;
    voiceBtn.classList.remove('listening');
    voiceBtn.innerHTML = '🎤';
    input.placeholder = "Ask anything...";
}

// Suggestion chip action
function useSuggestion(chip) {
    input.value = chip.textContent;
    sendMessage();
}

// Settings menu interactions
function toggleSettings() {
    playSound('click');
    settingsOverlay.classList.toggle('active');
}

function changeTheme(themeName) {
    document.body.setAttribute('data-theme', themeName);
}

// Copy raw code blocks helper
function copyCode(elementId) {
    const code = document.getElementById(elementId);
    if (!code) return;
    navigator.clipboard.writeText(code.innerText || code.textContent);
    playSound('click');
}

// Clear conversations helper
function clearChat() {
    chatContainer.innerHTML = `
        <div class="message-wrapper ai">
            <div class="message-card">
                <div class="message-bubble">Hello! 👋 Conversation cache cleared. I'm ready for new questions!</div>
                <div class="message-meta">
                    <span>Just now</span>
                </div>
            </div>
        </div>
    `;
    stopSpeaking();
    playSound('click');
    toggleSettings();
}

// Event Listeners
input.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

document.getElementById('speechSynthesisToggle').addEventListener('change', (e) => {
    useSpeechSynthesis = e.target.checked;
    if (!useSpeechSynthesis) stopSpeaking();
});

// Setup on load
window.addEventListener('DOMContentLoaded', () => {
    setupSpeechRecognition();
    
    // Set theme base on system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        document.getElementById('themeSelect').value = 'light';
        changeTheme('light');
    }
});
