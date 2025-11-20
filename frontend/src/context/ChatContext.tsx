import React, { createContext, useContext, ReactNode, useMemo } from "react";
import { useAppDispatch, useAppSelector } from "../store/hooks";
import {
  addMessage,
  startStreaming,
  appendStreamingContent,
  stopStreaming,
  updateMessageStatus,
  updateMessageContent,
  setError,
  setConversationId,
  setMessages,
  clearMessages,
  resetChatState,
  setMessageDraft,
} from "../store/slice/chatSlice";
import {
  useGetConversationsQuery,
  useLazyGetMessagesQuery,
  useSendMessageMutation,
  useStreamMessageMutation,
} from "../store/api/chatApi";
import { Message, Role, Conversation } from "../types";
import { useNavigate } from "react-router-dom";

interface ChatContextType {
  conversationId: string | null;
  isNewConversation: boolean;
  currentConversation: Conversation | null;
  messages: Message[];
  conversations: Conversation[] | undefined;
  isLoadingConversations: boolean;
  isSending: boolean;
  isStreaming: boolean;
  messageDraft: string;
  error: string | null;
  sendMessage: (
    content: string,
    sender?: Role,
    navigate?: (path: string) => void
  ) => void;
  startNewConversation: () => void;
  selectConversation: (id: string) => void;
  setDraft: (text: string) => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider = ({ children }: { children: ReactNode }) => {
  const dispatch = useAppDispatch();

  // Redux state
  const conversationId = useAppSelector((state) => state.chat.conversationId);
  const isNewConversation = useAppSelector(
    (state) => state.chat.isNewConversation
  );
  const isStreaming = useAppSelector((state) => state.chat.isStreaming);
  const messageDraft = useAppSelector((state) => state.chat.messageDraft);
  const error = useAppSelector((state) => state.chat.error);
  const messages = useAppSelector((state) => state.chat.messages);

  // API hooks
  const [sendMessageApi, { isLoading: isSending }] = useSendMessageMutation();
  const [streamMessageApi] = useStreamMessageMutation();
  const [fetchMessages] = useLazyGetMessagesQuery();
  const {
    data: conversations,
    isLoading: isLoadingConversations,
    refetch: refetchConversations,
  } = useGetConversationsQuery();

  // Derive current conversation object
  const currentConversation = useMemo(() => {
    if (!conversationId || !conversations) return null;
    return conversations.find((c) => c.id === conversationId) ?? null;
  }, [conversationId, conversations]);

  // Start a new conversation
  const startNewConversation = () => {
    dispatch(resetChatState());
  };

  // Select a conversation and load its messages
  const selectConversation = async (id: string) => {
    dispatch(setConversationId(id));

    try {
      const messages = await fetchMessages(id).unwrap();
      dispatch(setMessages(messages));
      dispatch(setError(null));
    } catch (err) {
      console.error("Failed to fetch messages:", err);
      dispatch(setError("Failed to load conversation"));
    }
  };

  // Send a message (non-streaming)
  const sendMessage = async (
    content: string,
    sender: Role = "user",
    navigate?: (path: string) => void
  ) => {
    const userMessageId = crypto.randomUUID();
    const streamingId = crypto.randomUUID();

    // Add user message
    const userMessage: Message = {
      id: userMessageId,
      sender,
      content,
      timestamp: Date.now(),
      status: "pending",
    };
    dispatch(addMessage(userMessage));

    // Add placeholder AI message
    const aiPlaceholder: Message = {
      id: streamingId,
      sender: "assistant",
      content: "",
      timestamp: Date.now(),
      status: "pending",
    };
    dispatch(addMessage(aiPlaceholder));

    dispatch(setError(null));

    try {
      const response = await sendMessageApi({
        question: content,
        conversationId: conversationId ?? undefined,
      }).unwrap();

      const result = response.data;

      // Set conversation ID if new
      if (!conversationId && result.conversation_id) {
        dispatch(setConversationId(result.conversation_id));
        refetchConversations();
      }
      if (isNewConversation && navigate) 
        navigate(`/chat/${result.conversation_id}`);

      // Update placeholder AI message with real content
      dispatch(
        updateMessageContent({
          id: streamingId,
          content: result.content,
          status: "sent",
          timestamp: new Date(result.timestamp).getTime(),
          conversationId: result.conversation_id,
        })
      );

      // Mark user message as sent
      dispatch(updateMessageStatus({ id: userMessageId, status: "sent" }));
    } catch (err) {
      console.error("Error sending message:", err);
      dispatch(setError("Message failed to send"));
      dispatch(updateMessageStatus({ id: userMessageId, status: "error" }));
      dispatch(updateMessageStatus({ id: streamingId, status: "error" }));
    }
  };

  // Update draft text
  const setDraft = (text: string) => {
    dispatch(setMessageDraft(text));
  };

  return (
    <ChatContext.Provider
      value={{
        conversationId,
        isNewConversation,
        currentConversation,
        conversations,
        isLoadingConversations,
        isSending,
        messages,
        isStreaming,
        messageDraft,
        error,
        sendMessage,
        startNewConversation,
        selectConversation,
        setDraft,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export const useChatContext = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return context;
};
