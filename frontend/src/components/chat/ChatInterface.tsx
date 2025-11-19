import React, { useEffect, useRef } from "react";
import ChatInput from "./ChatInput";
import NewChat from "../../pages/NewChatPage";
import MessageBubble from "./MessageBubble";
import { useChatContext } from "../../context/ChatContext";
import { useParams } from "react-router-dom";

export const ChatInterface: React.FC = () => {
  const {
    currentConversation,
    messages,
    isNewConversation,
    isStreaming,
    isSending,
    messageDraft,
    setDraft,
    sendMessage,
    selectConversation,
  } = useChatContext();
  const { conversationId } = useParams();

  useEffect(() => {
    if (conversationId) {
      selectConversation(conversationId); // fetch messages, set active
    }
  }, [conversationId]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, messages.at(-1)?.content]);

  const isEmpty = !currentConversation || messages.length === 0;

  return (
    <section className="relative flex flex-col h-[90vh] pb-10">
      <div className="flex-1 p-4 sm:p-6 overflow-y-auto scrollbar-thin">
        <div className="space-y-6 flex flex-col items-center justify-end max-w-[800px] m-auto mt-10">
          {messages.map((msg, idx) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              isSending={isSending}
              isStreaming={isStreaming}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <ChatInput
        value={messageDraft}
        onChange={setDraft}
        onSendMessage={sendMessage}
        isLoading={isSending || isStreaming}
      />
    </section>
  );
};
