import React from "react";
import ChatInput from "./ChatInput";
import { useChatContext } from "../../context/ChatContext";

const NewChat = () => {
  const {
    currentConversation,
    messages,
    isNewConversation,
    isStreaming,
    isSending,
    messageDraft,
    setDraft,
    sendMessage,
  } = useChatContext();

  return (
    <section className="flex flex-col justify-center pb-10 max-w-[900px] m-auto h-full pb-50 h-full overflow-hidden">
      <div className="flex flex-col items-center justify-center text-center my-10 w-full">
        <div className="text-4xl mb-2">
          <img src="favicon-32x32.png" alt="logo" className="w-15 h-15" />
        </div>
        <h2 className="text-2xl font-medium mb-2">Welcome to DAxent</h2>
        <div className="empty-subtitle">
          Upload well reports, ask questions, and get instant analysis with
          production estimates and nodal analysis.
        </div>
        <div className="flex lg:flex-row flex-col text-xs gap-4 justify-center items-center mt-4">
          <div className="bg-white dark:bg-gray-600 p-2 rounded-lg shadow-sm cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700">
            📄 Upload a well report
          </div>
          <div className="bg-white dark:bg-gray-600 p-2 rounded-lg shadow-sm cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700">
            📊 Analyze production rate
          </div>
          <div className="bg-white dark:bg-gray-600 p-2 rounded-lg shadow-sm cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700">
            🔍 Extract parameters
          </div>
          <div className="bg-white dark:bg-gray-600 p-2 rounded-lg shadow-sm cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700">
            ❓ How does it work?
          </div>
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

export default NewChat;
