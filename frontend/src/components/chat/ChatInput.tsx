import { useEffect, useRef, type KeyboardEvent } from 'react';
import PaperAirplaneIcon from '../ui/icons/PaperAirplaneIcon';
import PlusIcon from '../ui/icons/PlusIcon';

interface ChatInputProps {
  value: string;
  onChange: (text: string) => void;
  onSendMessage: (content: string) => void;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSendMessage,
  isLoading,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = 200;
      textareaRef.current.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
    }
  }, [value]);

  const handleSubmit = (e?: React.FormEvent<HTMLFormElement>) => {
    e?.preventDefault();
    if (value.trim()) {
      onSendMessage(value);
      onChange(''); // Clear draft after sending
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border- border-gray-300 dark:border-gray-700/50 flex-">
      <div className="relative max-w-[820px] m-auto">
        <form
          onSubmit={handleSubmit}
          className="w-full scrollbar-thin shadow-lg border p-4 rounded-3xl flex flex-col items-center justify-between gap-4 border-gray-300 dark:border-gray-700/50"
        >
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message here..."
            rows={1}
            className="flex- scrollbar-thin bg-transparent dark:text-gray-200 placeholder-gray-400 resize-none focus:outline-none max-h-48 w-full"
            disabled={isLoading}
            style={{ overflowY: 'auto' }}
          />

          <div className="flex justify-between items-center gap-2 w-full">
            <div className="file-upload-area icon">
              <label htmlFor="document" className="upload-icon cursor-pointer">
                <PlusIcon />
              </label>
              <input type="file" id="document" className="hidden" />
            </div>

            <button
              type="submit"
              disabled={isLoading || !value.trim()}
              className="bg-blue-600 text-white rounded-lg p-2 h-9 w-9 flex items-center justify-center flex-shrink-0 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors duration-200 focus:ring-2 focus:ring-blue-500 focus:outline-none"
              aria-label="Send message"
            >
              <PaperAirplaneIcon />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInput;