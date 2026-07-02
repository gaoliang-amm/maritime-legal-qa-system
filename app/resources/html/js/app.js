/**
 * Maritime Legal QA System - Frontend Application
 * Professional RAG-powered maritime law assistant
 */

// ==================== Configuration ====================
const CONFIG = {
    API_BASE: location.origin.startsWith('http') ? location.origin : 'http://localhost:8001',
    STREAM_TIMEOUT: 120000,
    MAX_HISTORY: 50,
    TYPING_DELAY: 20
};

// ==================== State Management ====================
const state = {
    sessionId: null,
    isLoading: false,
    history: [],
    currentMessages: []
};

// ==================== DOM Elements ====================
const DOM = {
    sidebar: document.getElementById('sidebar'),
    menuBtn: document.getElementById('menuBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    historyList: document.getElementById('historyList'),
    chatContainer: document.getElementById('chatContainer'),
    welcome: document.getElementById('welcome'),
    messages: document.getElementById('messages'),
    inputField: document.getElementById('inputField'),
    sendBtn: document.getElementById('sendBtn')
};

// ==================== Initialization ====================
document.addEventListener('DOMContentLoaded', () => {
    initSession();
    loadHistory();
    bindEvents();
    autoResizeTextarea();
});

function initSession() {
    state.sessionId = localStorage.getItem('sessionId') || generateSessionId();
    localStorage.setItem('sessionId', state.sessionId);
}

function generateSessionId() {
    return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// ==================== Event Binding ====================
function bindEvents() {
    // Send button
    DOM.sendBtn.addEventListener('click', handleSend);

    // Input field
    DOM.inputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // New chat button
    DOM.newChatBtn.addEventListener('click', startNewChat);

    // Menu button (mobile)
    DOM.menuBtn.addEventListener('click', toggleSidebar);

    // Close sidebar when clicking outside (mobile)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && !DOM.sidebar.contains(e.target) && !DOM.menuBtn.contains(e.target)) {
            DOM.sidebar.classList.remove('open');
        }
    });
}

function autoResizeTextarea() {
    DOM.inputField.addEventListener('input', () => {
        DOM.inputField.style.height = 'auto';
        DOM.inputField.style.height = Math.min(DOM.inputField.scrollHeight, 120) + 'px';
    });
}

function toggleSidebar() {
    DOM.sidebar.classList.toggle('open');
}

// ==================== Message Handling ====================
async function handleSend() {
    const query = DOM.inputField.value.trim();
    if (!query || state.isLoading) return;

    // Clear input
    DOM.inputField.value = '';
    DOM.inputField.style.height = 'auto';

    // Hide welcome screen
    if (DOM.welcome) {
        DOM.welcome.style.display = 'none';
    }

    // Add user message
    appendUserMessage(query);

    // Save to history
    saveToHistory(query);

    // Show loading
    state.isLoading = true;
    DOM.sendBtn.disabled = true;
    const loadingEl = appendLoadingMessage();

    try {
        await streamQuery(query, loadingEl);
    } catch (error) {
        console.error('Query failed:', error);
        removeLoading(loadingEl);
        appendErrorMessage('查询失败，请稍后重试');
    } finally {
        state.isLoading = false;
        DOM.sendBtn.disabled = false;
    }
}

// ==================== SSE Streaming ====================
async function streamQuery(query, loadingEl) {
    // Send query request
    const response = await fetch(`${CONFIG.API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            session_id: state.sessionId,
            is_stream: true
        })
    });

    if (!response.ok) {
        throw new Error('Query request failed');
    }

    const data = await response.json();
    const sessionId = data.session_id;

    // Connect SSE stream
    const eventSource = new EventSource(`${CONFIG.API_BASE}/stream/${sessionId}`);

    let answer = '';
    let messageEl = null;
    let refs = [];
    let chunks = [];

    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            eventSource.close();
            reject(new Error('Request timeout'));
        }, CONFIG.STREAM_TIMEOUT);

        // Listen for progress events
        eventSource.addEventListener('progress', (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('[Progress]', data);
                updateProgress(loadingEl, data.done_list, data.running_list, data.status);
            } catch (e) {
                console.error('Parse progress error:', e);
            }
        });

        // Listen for delta events (streaming output)
        eventSource.addEventListener('delta', (event) => {
            try {
                const data = JSON.parse(event.data);
                if (!messageEl) {
                    removeLoading(loadingEl);
                    messageEl = appendAIMessage('', true);
                }
                answer += data.delta || '';
                updateMessageContent(messageEl, answer);
            } catch (e) {
                console.error('Parse delta error:', e);
            }
        });

        // Listen for final events (complete result)
        eventSource.addEventListener('final', (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('[DEBUG] Final event data:', data);
                console.log('[DEBUG] image_urls:', data.image_urls);
                clearTimeout(timeout);
                eventSource.close();

                // Update final answer
                if (messageEl) {
                    updateMessageContent(messageEl, data.answer || answer);
                    // Add sources if available
                    if (data.image_urls && data.image_urls.length > 0) {
                        appendSources(messageEl, data.image_urls);
                    }
                } else {
                    removeLoading(loadingEl);
                    appendAIMessage(data.answer || answer);
                }

                resolve();
            } catch (e) {
                console.error('Parse final error:', e);
                resolve();
            }
        });

        // Listen for error events
        eventSource.addEventListener('error', (event) => {
            try {
                clearTimeout(timeout);
                eventSource.close();

                if (event.data) {
                    const data = JSON.parse(event.data);
                    reject(new Error(data.message || 'Query failed'));
                } else {
                    if (answer) {
                        resolve();
                    } else {
                        reject(new Error('SSE connection failed'));
                    }
                }
            } catch (e) {
                reject(new Error('SSE connection failed'));
            }
        });
    });
}

// ==================== Message Rendering ====================
function appendUserMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-user';
    messageDiv.innerHTML = `
        <div class="message-bubble">${escapeHtml(content)}</div>
    `;
    DOM.messages.appendChild(messageDiv);
    scrollToBottom();
}

function appendAIMessage(content, isStreaming = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-ai';
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-body">
            <div class="message-answer">${isStreaming ? '' : formatContent(content)}</div>
        </div>
    `;
    DOM.messages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function updateMessageContent(messageEl, content) {
    const answerEl = messageEl.querySelector('.message-answer');
    if (answerEl) {
        answerEl.innerHTML = formatContent(content);
        scrollToBottom();
    }
}

function appendLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message-ai';
    loadingDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-body">
            <div class="message-loading">
                <div class="loading-spinner"></div>
                <div class="loading-text">
                    <span>正在初始化...</span>
                    <span class="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </span>
                </div>
            </div>
            <div class="progress-container" style="display: none;">
                <div class="progress-header">
                    <span class="progress-title">📊 处理进度</span>
                </div>
                <div class="progress-list"></div>
            </div>
        </div>
    `;
    DOM.messages.appendChild(loadingDiv);
    scrollToBottom();
    return loadingDiv;
}

function updateLoadingText(loadingEl, text) {
    const textEl = loadingEl.querySelector('.loading-text span:first-child');
    if (textEl) {
        textEl.textContent = text;
    }
}

function updateProgress(loadingEl, doneList, runningList, status) {
    const progressContainer = loadingEl.querySelector('.progress-container');
    const progressList = loadingEl.querySelector('.progress-list');

    if (!progressContainer || !progressList) return;

    // 显示进度容器
    progressContainer.style.display = 'block';

    // 更新加载文本
    const loadingText = loadingEl.querySelector('.loading-text span:first-child');
    if (runningList && runningList.length > 0) {
        loadingText.textContent = `正在处理: ${runningList[0]}`;
    } else if (status === 'completed') {
        loadingText.textContent = '处理完成';
    }

    // 构建进度列表
    let html = '';

    // 已完成节点
    if (doneList && doneList.length > 0) {
        doneList.forEach(item => {
            html += `<div class="progress-item done"><span class="progress-icon">✅</span><span>${item}</span></div>`;
        });
    }

    // 进行中节点
    if (runningList && runningList.length > 0) {
        runningList.forEach(item => {
            html += `<div class="progress-item running"><span class="progress-icon">⏳</span><span>${item}</span></div>`;
        });
    }

    progressList.innerHTML = html;
    scrollToBottom();
}

function removeLoading(loadingEl) {
    if (loadingEl && loadingEl.parentNode) {
        loadingEl.parentNode.removeChild(loadingEl);
    }
}

function appendErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message-ai';
    errorDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-body">
            <div class="message-error">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <span>${escapeHtml(message)}</span>
            </div>
        </div>
    `;
    DOM.messages.appendChild(errorDiv);
    scrollToBottom();
}

function appendSources(messageEl, urls) {
    if (!urls || urls.length === 0) return;

    const bodyEl = messageEl.querySelector('.message-body');
    if (!bodyEl) return;

    // 判断是否是图片 URL
    const isImageUrl = (url) => /\.(jpg|jpeg|png|gif|webp|bmp|svg)/i.test(url);

    const refsDiv = document.createElement('div');
    refsDiv.className = 'message-refs';
    refsDiv.innerHTML = `
        <div class="message-refs-header" onclick="toggleSection(this)">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
            <span>📚 参考来源 (${urls.length})</span>
        </div>
        <div class="message-refs-content">
            ${urls.map((url, index) => `
                <div class="ref-item">
                    ${isImageUrl(url) ? `
                        <div class="ref-image-container">
                            <img src="${escapeHtml(url)}" alt="参考图片 ${index + 1}" class="ref-image" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                            <div style="display:none; color: var(--text-tertiary); font-size: 12px;">图片加载失败</div>
                            <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="ref-link">${escapeHtml(url)}</a>
                        </div>
                    ` : `
                        <span class="ref-icon">📄</span>
                        <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="ref-link">${escapeHtml(url)}</a>
                    `}
                </div>
            `).join('')}
        </div>
    `;
    bodyEl.appendChild(refsDiv);
}

function appendChunks(messageEl, chunks) {
    if (!chunks || chunks.length === 0) return;

    const bodyEl = messageEl.querySelector('.message-body');
    if (!bodyEl) return;

    const contextDiv = document.createElement('div');
    contextDiv.className = 'message-context';
    contextDiv.innerHTML = `
        <div class="message-context-header" onclick="toggleSection(this)">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
            <span>🔎 检索证据 (${chunks.length})</span>
        </div>
        <div class="message-context-content">
            ${chunks.map((chunk, index) => `
                <div class="context-item" onmouseenter="highlightContext(this)" onmouseleave="unhighlightContext(this)">
                    <div class="context-item-header">
                        <span>Chunk ${index + 1}</span>
                        ${chunk.score ? `<span class="context-item-score">${(chunk.score * 100).toFixed(0)}%</span>` : ''}
                    </div>
                    <div>${escapeHtml(chunk.text || chunk.content || '')}</div>
                </div>
            `).join('')}
        </div>
    `;
    bodyEl.appendChild(contextDiv);
}

function toggleSection(header) {
    header.classList.toggle('expanded');
    const content = header.nextElementSibling;
    content.classList.toggle('show');
}

function highlightContext(element) {
    element.style.background = 'var(--primary-bg)';
    element.style.borderLeftColor = 'var(--primary-hover)';
}

function unhighlightContext(element) {
    element.style.background = '';
    element.style.borderLeftColor = '';
}

// ==================== History Management ====================
function saveToHistory(query) {
    const historyItem = {
        id: Date.now(),
        query: query,
        sessionId: state.sessionId,
        timestamp: new Date().toISOString()
    };

    state.history.unshift(historyItem);

    if (state.history.length > CONFIG.MAX_HISTORY) {
        state.history = state.history.slice(0, CONFIG.MAX_HISTORY);
    }

    localStorage.setItem('chatHistory', JSON.stringify(state.history));
    renderHistory();
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('chatHistory');
        if (saved) {
            state.history = JSON.parse(saved);
        }
    } catch (e) {
        console.error('Load history failed:', e);
    }
    renderHistory();
}

function renderHistory() {
    if (!DOM.historyList) return;

    if (state.history.length === 0) {
        DOM.historyList.innerHTML = '<div class="history-empty" style="padding: 16px; text-align: center; color: var(--text-tertiary); font-size: 13px;">暂无对话记录</div>';
        return;
    }

    DOM.historyList.innerHTML = state.history.map(item => `
        <div class="history-item ${item.sessionId === state.sessionId ? 'active' : ''}"
             data-id="${item.id}"
             onclick="loadHistoryItem(${item.id})">
            <span class="history-item-icon">💬</span>
            <span class="history-item-text">${escapeHtml(item.query)}</span>
            <span class="history-item-time">${formatTime(item.timestamp)}</span>
        </div>
    `).join('');
}

function loadHistoryItem(id) {
    const item = state.history.find(h => h.id === id);
    if (!item) return;

    state.sessionId = item.sessionId;
    localStorage.setItem('sessionId', state.sessionId);

    DOM.messages.innerHTML = '';

    if (DOM.welcome) {
        DOM.welcome.style.display = 'none';
    }

    appendUserMessage(item.query);
    appendAIMessage('正在加载历史对话...');

    renderHistory();

    if (window.innerWidth <= 768) {
        DOM.sidebar.classList.remove('open');
    }
}

function startNewChat() {
    state.sessionId = generateSessionId();
    localStorage.setItem('sessionId', state.sessionId);

    DOM.messages.innerHTML = '';

    if (DOM.welcome) {
        DOM.welcome.style.display = 'flex';
    }

    renderHistory();

    if (window.innerWidth <= 768) {
        DOM.sidebar.classList.remove('open');
    }
}

// ==================== Example Questions ====================
function askExample(button) {
    const question = button.getAttribute('data-question');
    if (question) {
        DOM.inputField.value = question;
        handleSend();
    }
}

// ==================== Utility Functions ====================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatContent(content) {
    if (!content) return '';

    // Simple markdown rendering
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/`(.*?)`/g, '<code style="background: var(--bg-tertiary); padding: 2px 4px; border-radius: 3px; font-size: 13px;">$1</code>');
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前';
    if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前';

    return date.toLocaleDateString('zh-CN', {
        month: 'numeric',
        day: 'numeric'
    });
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        DOM.chatContainer.scrollTop = DOM.chatContainer.scrollHeight;
    });
}
