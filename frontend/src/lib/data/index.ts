import type { Message, Role } from "../../types"; // Import Message

// Helper function to generate unique IDs
const generateId = (): string => Math.random().toString(36).substring(2, 15);

// Helper function to generate timestamps
const generateTimestamp = (): string => new Date().toISOString();

// Mock messages for a conversation
const createMockMessages = (
  conversationId: string,
  count: number
): Message[] => {
  const messages: Message[] = [];
  for (let i = 0; i < count; i++) {
    messages.push({
      id: generateId(),
      content: `This is message number ${
        i + 1
      } in conversation ${conversationId}.`,
      sender: i % 2 === 0 ? "assistant" : "user",
      timestamp: generateTimestamp(),
      conversationId: conversationId,
    });
  }
  return messages;
};

// Mock conversations
interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

const mockConversations: Conversation[] = [
  {
    id: generateId(),
    title: "Initial Chat",
    messages: createMockMessages(generateId(), 5),
  },
  {
    id: generateId(),
    title: "Follow-up",
    messages: createMockMessages(generateId(), 8),
  },
  {
    id: generateId(),
    title: "Support Request",
    messages: createMockMessages(generateId(), 3),
  },
];

// Exporting the mock conversations
export { mockConversations };
