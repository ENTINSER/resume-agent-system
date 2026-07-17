const API_BASE = '';
let currentTaskId = null;
let eventSource = null;

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast-hide');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function setButtonsLoading(loading) {
    ['btn-approve', 'btn-rewrite', 'btn-evaluate', 'btn-train'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = loading;
    });
}

function startTask() {
    const input = document.getElementById('task-input');
    const requirements = input.value.trim();
    if (!requirements) {
        showToast('请输入项目需求', 'warning');
        return;
    }

    fetch(`${API_BASE}/api/v1/tasks`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({requirements})
    })
    .then(r => r.json())
    .then(data => {
        if (data.task_id) {
            currentTaskId = data.task_id;
            input.value = '';
            showToast(`任务已创建: ${currentTaskId.slice(0, 8)}`, 'success');
            connectSSE(currentTaskId);
            updateUI('running', {phase: 'pending'});
            loadHistory();
        } else {
            showToast('创建失败: ' + (data.error || '未知错误'), 'error');
        }
    })
    .catch(err => showToast('请求失败: ' + err, 'error'));
}

function connectSSE(taskId) {
    if (eventSource) eventSource.close();

    eventSource = new EventSource(`${API_BASE}/api/v1/tasks/${taskId}/events`);
    const logEl = document.getElementById('event-log');
    if (logEl) logEl.textContent = '';

    eventSource.onmessage = (e) => {
        const event = JSON.parse(e.data);
        appendLog(JSON.stringify(event, null, 2));
        const status = event.status || event.current_phase;

        if (event.current_phase === 'human_review') {
            updateUI('pending', event);
        } else if (status === 'completed') {
            updateUI('completed', event);
            loadTaskResult(taskId);
            eventSource.close();
        } else if (status === 'failed') {
            updateUI('completed', event);
            showToast('任务失败: ' + (event.error || '未知错误'), 'error');
            eventSource.close();
        } else {
            updateUI('running', event);
        }
    };

    eventSource.onerror = () => {
        console.warn('SSE 连接中断');
    };
}

function appendLog(text) {
    const logEl = document.getElementById('event-log');
    if (!logEl) return;
    logEl.textContent += text + '\n';
    logEl.scrollTop = logEl.scrollHeight;
}

function updateUI(view, event) {
    document.getElementById('idle-section').classList.add('hidden');
    document.getElementById('running-section').classList.add('hidden');
    document.getElementById('pending-section').classList.add('hidden');
    document.getElementById('completed-section').classList.add('hidden');

    if (view === 'running') {
        document.getElementById('running-section').classList.remove('hidden');
        document.getElementById('task-message').textContent = event.current_phase || '运行中...';
    } else if (view === 'pending') {
        document.getElementById('pending-section').classList.remove('hidden');
        document.getElementById('task-id').textContent = currentTaskId?.slice(0, 8) || '-';
        document.getElementById('current-phase').textContent = event.current_phase || '-';
        document.getElementById('task-status-text').textContent = event.status || 'human_review';
        setButtonsLoading(false);
    } else if (view === 'completed') {
        document.getElementById('completed-section').classList.remove('hidden');
    }
}

function respond(feedback) {
    if (!currentTaskId) return;
    setButtonsLoading(true);

    fetch(`${API_BASE}/api/v1/tasks/${currentTaskId}/review`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({feedback})
    })
    .then(r => r.json())
    .then(data => {
        showToast('反馈已提交', 'success');
        updateUI('running', {current_phase: '继续执行...'});
    })
    .catch(err => {
        showToast('提交失败: ' + err, 'error');
        setButtonsLoading(false);
    });
}

function loadTaskResult(taskId) {
    fetch(`${API_BASE}/api/v1/tasks/${taskId}`)
    .then(r => r.json())
    .then(data => {
        const resultEl = document.getElementById('result-json');
        if (resultEl) resultEl.textContent = JSON.stringify(data, null, 2);
    });
}

function resetUI() {
    currentTaskId = null;
    if (eventSource) eventSource.close();
    document.getElementById('idle-section').classList.remove('hidden');
    document.getElementById('running-section').classList.add('hidden');
    document.getElementById('pending-section').classList.add('hidden');
    document.getElementById('completed-section').classList.add('hidden');
    loadHistory();
}

function loadHistory() {
    fetch(`${API_BASE}/api/v1/tasks`)
    .then(r => r.json())
    .then(data => {
        const tbody = document.getElementById('history-body');
        const tasks = data.tasks || [];
        if (tasks.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center">暂无任务</td></tr>';
            return;
        }
        tbody.innerHTML = tasks.map(t => `
            <tr>
                <td>${t.task_id?.slice(0, 8) || '-'}</td>
                <td><span class="badge badge-${t.status === 'completed' ? 'success' : t.status === 'failed' ? 'danger' : 'secondary'}">${t.status}</span></td>
                <td>${t.current_phase || '-'}</td>
                <td>${t.created_at ? new Date(t.created_at * 1000).toLocaleString() : '-'}</td>
            </tr>
        `).join('');
    })
    .catch(err => console.error('加载历史失败:', err));
}

loadHistory();
