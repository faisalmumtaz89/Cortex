// What is typed in the prompt input, read as each key arrives: the global key
// handler needs it to tell Enter on a permission prompt from Enter that ends
// a message (a store value updated on content change can lag a fast burst).

let read: () => string = () => ""

export function registerPromptDraft(reader: () => string): void {
  read = reader
}

export function promptHasDraft(): boolean {
  return read().trim().length > 0
}
