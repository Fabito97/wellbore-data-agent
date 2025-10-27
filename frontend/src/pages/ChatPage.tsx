// import React from 'react'
import Header from "../components/layout/Header";
import { ChatInterface } from "../features/chat/ChatInterface";

const ChatPage = () => {
  return (
    <div className="relative h-full">
      <Header />

      {/* Chat Panel */}
      <ChatInterface />
    </div>
  );
};

export default ChatPage;
