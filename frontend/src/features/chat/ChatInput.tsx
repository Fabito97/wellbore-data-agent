// import React from "react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import PaperAirplaneIcon from "../../components/icons/PaperAirplaneIcon";

interface ChatInputProps {
  onSendMessage: (message: string, file?: File) => void;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = 200; // 5 lines of text roughly
      textareaRef.current.style.height = `${Math.min(
        scrollHeight,
        maxHeight
      )}px`;
    }
  }, [inputValue]);

  const handleSubmit = (e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gray-300 dark:border-gray-700/50 flex-1 px-5 py-10">
      <div className="position relative max-w-[900px] m-auto">
        <div className="file-upload-area hidden">
          <div className="upload-icon">📁</div>
          <div className="upload-text">
            <strong>Drop files here</strong> or click to browse
            <div style={{ fontSize: "12px", marginTop: " 4px" }}>
              PDF, DOCX (max 50MB)
            </div>
          </div>
        </div>

        <form className="w-full border p-4 rounded-3xl flex items-end justify-between gap-4 border-gray-300 dark:border-gray-700/50">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message here..."
            rows={1}
            className="flex-1 bg-transparent dark:text-gray-200 placeholder-gray-400 resize-none focus:outline-none max-h-48"
            disabled={isLoading}
            style={{ overflowY: "auto" }}
          />

          {/* <button
            className="text-4xl p-3 h-10 w-10 flex items-center justify-center rounded-full"
            disabled
          >
            ➤
          </button> */}
          <button
            type="submit"
            disabled={isLoading || !inputValue.trim()}
            className="bg-blue-600 text-white rounded-lg p-2 h-10 w-10 flex items-center justify-center flex-shrink-0 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors duration-200 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            aria-label="Send message"
          >
            <PaperAirplaneIcon />
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInput;
