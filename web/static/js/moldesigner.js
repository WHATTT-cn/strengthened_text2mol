function designDrug() {
    const textInput = document.getElementById('textInput');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');

    if (!textInput.value.trim()) {
        showError('请输入药物分子的性质');
        return;
    }

    // 显示加载动画
    loadingDiv.style.display = 'block';
    resultsDiv.innerHTML = '';

    fetch('/moldesigner', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({properties: textInput.value})
    })
    .then(response => response.json())
    .then(data => {
        loadingDiv.style.display = 'none';
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data.results);
        }
    })
    .catch(error => {
        loadingDiv.style.display = 'none';
        showError('设计过程中发生错误，请稍后重试');
        console.error('Error:', error);
    });
}

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (!results || results.length === 0) {
        resultsDiv.innerHTML = '<div class="alert alert-info">未找到相关设计建议</div>';
        return;
    }

    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'card molecule-card mb-3';
        card.innerHTML = `
            <div class="card-body">
                <p class="card-text">${result.suggestion}</p>
            </div>
        `;
        resultsDiv.appendChild(card);
    });
}

function showError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="error-message">
            ${message}
        </div>
    `;
}