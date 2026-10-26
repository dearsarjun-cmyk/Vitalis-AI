(() => {
  const $ = (s) => document.querySelector(s);
  const form = $('#chatForm'), input = $('#promptInput'), send = $('#sendButton');
  const messages = $('#messages'), welcome = $('#welcome'), scroll = $('#chatScroll');
  const statusLabel = $('#connectionLabel'), sideStatus = $('#sideModelStatus');
  const noticeText = $('#noticeText'), noticeDot = document.querySelector('.notice-dot');
  const historyList = $('#historyList'), toast = $('#toast');
  let ready = false, busy = false, currentId = null, conversations = [];
  try { conversations = JSON.parse(localStorage.getItem('vitalis_chats') || '[]'); } catch (_) { conversations = []; }
  const save = () => { try { localStorage.setItem('vitalis_chats', JSON.stringify(conversations.slice(0, 15))); } catch (_) {} };
  function showToast(text) { toast.textContent = text; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2400); }
  function scrollBottom() { scroll.scrollTo({ top: scroll.scrollHeight, behavior: 'smooth' }); }
  function setStatus(s) {
    ready = !!s.ready;
    statusLabel.textContent = ready ? 'Model online' : 'Model needs training';
    sideStatus.textContent = ready ? 'Ready for prompts' : 'No trained checkpoint';
    noticeText.textContent = ready ? 'VITALIS Core is ready · Experimental model' : 'The website is running, but a trained model checkpoint has not been found.';
    noticeDot.classList.toggle('ready', ready);
    document.querySelector('.model-orb').style.background = ready ? '#6cd5b3' : '#e5b56a';
    document.querySelector('.model-orb').style.boxShadow = ready ? '0 0 10px #6cd5b355' : '0 0 10px #e5b56a55';
  }
  async function checkStatus() {
    try { const r = await fetch('/status'); setStatus(await r.json()); }
    catch (_) { statusLabel.textContent = 'Server unavailable'; sideStatus.textContent = 'Connection failed'; noticeText.textContent = 'Could not connect to the VITALIS server.'; }
  }
  function addMessage(role, text, typing = false) {
    const row = document.createElement('article'); row.className = `message ${role}`;
    const avatar = document.createElement('div'); avatar.className = 'message-avatar'; avatar.textContent = role === 'user' ? 'Y' : 'V';
    const content = document.createElement('div'); content.className = 'message-content';
    const name = document.createElement('div'); name.className = 'message-name'; name.textContent = role === 'user' ? 'You' : 'VITALIS';
    const body = document.createElement('div'); body.className = 'message-text';
    if (typing) { body.innerHTML = '<span class="typing-dots"><i></i><i></i><i></i></span>'; }
    else body.textContent = text;
    content.append(name, body); row.append(avatar, content); messages.append(row); welcome.hidden = true; scrollBottom();
    return body;
  }
  function renderHistory() {
    historyList.replaceChildren();
    if (!conversations.length) { const empty = document.createElement('div'); empty.className = 'empty-history'; empty.textContent = 'Your conversations will appear here.'; historyList.append(empty); return; }
    conversations.slice(0, 8).forEach(c => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'history-item';
      const icon = document.createElement('span'); icon.textContent = '◌'; icon.style.color = '#8c9cff';
      const title = document.createElement('span'); title.textContent = c.title || 'New conversation';
      b.append(icon, title); b.title = title.textContent;
      b.addEventListener('click', () => openConversation(c.id)); historyList.append(b);
    });
  }
  function persistCurrent() {
    if (!currentId) return;
    const existing = conversations.find(c => c.id === currentId);
    const transcript = [...messages.querySelectorAll('.message')].map(m => ({ role: m.classList.contains('user') ? 'user' : 'assistant', text: m.querySelector('.message-text')?.textContent || '' })).filter(m => m.text.trim());
    if (!transcript.length) return;
    const title = transcript.find(m => m.role === 'user')?.text.slice(0, 42) || 'New conversation';
    if (existing) { existing.messages = transcript; existing.title = title; existing.updated = Date.now(); }
    else conversations.unshift({ id: currentId, title, messages: transcript, updated: Date.now() });
    conversations.sort((a,b) => b.updated - a.updated); conversations = conversations.slice(0, 15); save(); renderHistory();
  }
  function newChat() { currentId = null; messages.replaceChildren(); welcome.hidden = false; input.value = ''; input.focus(); }
  function openConversation(id) {
    const c = conversations.find(x => x.id === id); if (!c) return;
    currentId = id; messages.replaceChildren(); welcome.hidden = true;
    c.messages.forEach(m => addMessage(m.role, m.text)); $('#sidebar').classList.remove('open');
  }
  async function sendMessage(raw) {
    const prompt = raw.trim(); if (!prompt || busy) return;
    if (!currentId) currentId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()));
    addMessage('user', prompt); input.value = ''; input.style.height = 'auto'; busy = true; send.disabled = true; send.querySelector('span:first-child').textContent = 'Thinking';
    const answer = addMessage('assistant', '', true);
    try {
      const response = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
      answer.textContent = data.reply || 'The model generated no text. Try another prompt.';
    } catch (err) { answer.textContent = err.message || 'Could not reach the VITALIS server.'; }
    finally { busy = false; send.disabled = false; send.querySelector('span:first-child').textContent = 'Send'; input.focus(); persistCurrent(); scrollBottom(); }
  }
  form.addEventListener('submit', e => { e.preventDefault(); sendMessage(input.value); });
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 150)}px`; });
  document.querySelectorAll('[data-prompt]').forEach(b => b.addEventListener('click', () => { input.value = b.dataset.prompt; input.focus(); input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight,150)}px`; }));
  $('#newChat').addEventListener('click', newChat);
  $('#clearChat').addEventListener('click', () => { newChat(); showToast('Conversation cleared'); });
  $('#mobileMenu').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
  document.addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); newChat(); } if (e.key === 'Escape') $('#sidebar').classList.remove('open'); });
  renderHistory(); checkStatus();
})();
