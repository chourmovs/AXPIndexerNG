import {openDocument, openDocumentDirectory} from './api.js';

export function createDocumentActions(documentId) {
  const container = document.createElement('div'); container.className = 'document-actions';
  const error = document.createElement('span'); error.className = 'action-error'; error.setAttribute('role', 'status');
  for (const [label, action] of [['Open file', openDocument], ['Open dir', openDocumentDirectory]]) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'secondary compact'; button.textContent = label;
    button.addEventListener('click', async () => { error.textContent = ''; button.disabled = true;
      try { await action(documentId); } catch (exception) { error.textContent = exception.message; } finally { button.disabled = false; }
    }); container.append(button);
  }
  container.append(error); return container;
}
