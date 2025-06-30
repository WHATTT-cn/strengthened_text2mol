// 聊天历史记录
let chatHistory = [];

// 发送消息
function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // 添加用户消息到聊天界面
    addMessage(message, 'user');
    
    // 清空输入框
    messageInput.value = '';
    
    // 显示AI正在输入指示器
    showTypingIndicator();
    
    // 发送消息到后端
    fetch('/chat_with_ai', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            history: chatHistory
        })
    })
    .then(response => response.json())
    .then(data => {
        hideTypingIndicator();
        
        if (data.error) {
            addMessage('抱歉，我遇到了一些问题：' + data.error, 'assistant');
        } else {
            addMessage(data.response, 'assistant');
        }
    })
    .catch(error => {
        hideTypingIndicator();
        addMessage('抱歉，网络连接出现问题，请稍后重试。', 'assistant');
        console.error('Error:', error);
    });
}

// 添加消息到聊天界面
function addMessage(content, sender) {
    const chatContainer = document.getElementById('chatContainer');
    
    // 创建消息元素
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? '我' : 'AI';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = formatMessage(content);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    // 添加到聊天容器
    chatContainer.appendChild(messageDiv);
    
    // 滚动到底部
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // 保存到历史记录
    chatHistory.push({
        role: sender === 'user' ? 'user' : 'assistant',
        content: content
    });
}

// 格式化消息内容（支持换行和代码块）
function formatMessage(content) {
    // 将换行符转换为<br>标签
    content = content.replace(/\n/g, '<br>');
    
    // 检测并格式化代码块
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
        return `<pre><code class="language-${lang || 'text'}">${code}</code></pre>`;
    });
    
    // 检测并格式化行内代码
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    return content;
}

// 显示输入指示器
function showTypingIndicator() {
    const chatContainer = document.getElementById('chatContainer');
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';
    
    const typingContent = document.createElement('div');
    typingContent.className = 'message-content';
    typingContent.innerHTML = '<span class="typing-dots">AI正在思考</span>';
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(typingContent);
    
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 隐藏输入指示器
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// 清空聊天记录
function clearChat() {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = `
        <div class="message assistant">
            <div class="message-avatar">AI</div>
            <div class="message-content">
                您好！我是您的智慧药物设计助手。我可以帮助您：
                <br>• 分析分子结构和性质
                <br>• 提供药物设计建议
                <br>• 解答化学相关问题
                <br>• 协助分子优化
                <br><br>请告诉我您需要什么帮助？
            </div>
        </div>
    `;
    
    chatHistory = [];
}

// 回车键发送消息
document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 自动聚焦到输入框
    messageInput.focus();
});