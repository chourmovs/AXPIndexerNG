export function appendHighlighted(element, value, terms) {
  const text = String(value || '');
  if (!terms.length) { element.append(document.createTextNode(text)); return; }
  const fold = value => String(value).normalize('NFKD').replace(/\p{M}/gu, '').toLocaleLowerCase();
  let folded = ''; const starts = []; const ends = [];
  for (let index = 0; index < text.length;) { const point = text.codePointAt(index); const original = String.fromCodePoint(point);
    const next = index + original.length; const normalized = fold(original); folded += normalized;
    for (let count = 0; count < normalized.length; count += 1) { starts.push(index); ends.push(next); } index = next; }
  const escaped = terms.map(fold).filter(Boolean).map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  if (!escaped.length) { element.append(document.createTextNode(text)); return; }
  const matcher = new RegExp(`(${escaped.join('|')})`, 'gu'); let offset = 0;
  for (const match of folded.matchAll(matcher)) { const start = starts[match.index]; const end = ends[match.index + match[0].length - 1];
    element.append(document.createTextNode(text.slice(offset, start)));
    const mark = document.createElement('mark'); mark.textContent = text.slice(start, end); element.append(mark); offset = end; }
  element.append(document.createTextNode(text.slice(offset)));
}
export function element(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className;
  if (text != null) node.textContent = text; return node; }
