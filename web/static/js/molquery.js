function queryMolecule() {
    const textInput = document.getElementById('textInput');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');

    if (!textInput.value.trim()) {
        showError('请输入分子ID或标准命名');
        return;
    }

    // 显示加载动画
    loadingDiv.style.display = 'block';
    resultsDiv.innerHTML = '';

    fetch('/molquery', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: textInput.value})
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
        showError('搜索过程中发生错误，请稍后重试');
        console.error('Error:', error);
    });
}

function displayResults(results) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (!results || results.length === 0) {
        resultsDiv.innerHTML = '<div class="alert alert-info">未找到匹配的分子</div>';
        return;
    }

    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'card molecule-card mb-3';
        card.innerHTML = `
            <div class="card-body">
                <h5 class="card-title">分子ID: ${result.id}</h5>
                <p class="card-text">${result.description}</p>
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