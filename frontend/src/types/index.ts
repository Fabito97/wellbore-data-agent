export type Role = "user" | "assistant";

export type status = "pending" | "sent" | "error" | "streaming";

export interface Message {
  id: string;
  content: string;
  sender: Role;
  timestamp: number; // ISO 8601 format
  conversationId?: string;
  isStreaming?: boolean;
  status?: status;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  isCurrent?: boolean;
  createdAt?: string;
  updatedAt?: string;
}

export interface MessageRequest {
  userMessage: string;
  conversation_id?: string;
  documentIds?: string[];
}
