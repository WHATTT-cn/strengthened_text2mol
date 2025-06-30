function searchMolecules() {
    const textInput = document.getElementById('textInput').value.trim();
    if (!textInput) {
        alert('请输入分子ID或标准命名');
        return;
    }

    // 显示加载动画
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    fetch('/molquery', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            input: textInput
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        if (data.status === 'success') {
            displayResults(data.results, 'search');
        } else {
            document.getElementById('results').innerHTML = `
                <div class="alert alert-danger" role="alert">
                    ${data.error || '搜索失败'}
                </div>
            `;
        }
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').innerHTML = `
            <div class="alert alert-danger" role="alert">
                网络错误: ${error.message}
            </div>
        `;
    });
}

function generateCaption() {
    const textInput = document.getElementById('textInput').value.trim();
    if (!textInput) {
        alert('请输入分子ID或标准命名');
        return;
    }

    // 显示加载动画
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    fetch('/molecule_captioning', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            molecule_input: textInput
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        
        if (data.status === 'success') {
            displayResults(data.results, 'caption');
        } else {
            document.getElementById('results').innerHTML = `
                <div class="alert alert-danger" role="alert">
                    ${data.error || '生成描述失败'}
                </div>
            `;
        }
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').innerHTML = `
            <div class="alert alert-danger" role="alert">
                网络错误: ${error.message}
            </div>
        `;
    });
}

function displayResults(results, type) {
    const resultsDiv = document.getElementById('results');
    
    if (type === 'search') {
        // 显示搜索结果
        let html = '<h5>搜索结果：</h5>';
        results.forEach((result, index) => {
            html += `
                <div class="molecule-card">
                    <h6>分子ID: ${result.id}</h6>
                    <p><strong>描述:</strong> ${result.description}</p>
                </div>
            `;
        });
        resultsDiv.innerHTML = html;
    } else if (type === 'caption') {
        // 显示captioning结果
        let html = '<h5>分子描述生成结果：</h5>';
        results.forEach((result, index) => {
            html += `
                <div class="molecule-card">
                    <h6>分子ID: ${result.id}</h6>
                    <p><strong>原始描述:</strong> ${result.original_description}</p>
                    <p><strong>生成描述:</strong> <span class="text-success">${result.generated_caption}</span></p>
                </div>
            `;
        });
        resultsDiv.innerHTML = html;
    }
}

// 添加回车键支持
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    textInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            searchMolecules();
        }
    });
});