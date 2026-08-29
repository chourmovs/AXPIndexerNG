export function appendHighlighted(element, value, terms) {
  const text = String(value || '');
  if (!terms.length) { element.append(document.createTextNode(text)); return; }
  const escaped = terms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const matcher = new RegExp(`(${escaped.join('|')})`, 'giu'); let offset = 0;
  for (const match of text.matchAll(matcher)) { element.append(document.createTextNode(text.slice(offset, match.index)));
    const mark = document.createElement('mark'); mark.textContent = match[0]; element.append(mark); offset = match.index + match[0].length; }
  element.append(document.createTextNode(text.slice(offset)));
}
export function element(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className;
  if (text != null) node.textContent = text; return node; }
