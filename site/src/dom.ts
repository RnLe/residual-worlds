// DOM helper.

export function el<K extends keyof HTMLElementTagNameMap>(
  name: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(name);
  if (className !== undefined) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
