import { useMemo } from "react";
import LoadingSpinner from "../ui/LoadingSpinner";
import UserIcon from "../ui/icons/UserIcon";
import LogoIcon from "../ui/icons/LogoIcon";
import type { Message } from "../../types";
import DOMPurify from "dompurify";
import { marked } from "marked";

interface MessageBubbleProps {
  message: Message;
  isSending: boolean;
  isStreaming: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isSending,
  isStreaming,
}) => {
  const isUser = message.sender === "user";
  const isAI = message.sender === "assistant";

  const bubbleAlignment = isUser ? "justify-end" : "justify-start";
  const bubbleColor = isUser ? "dark:bg-gray-700/50 bg-gray-200" : "bg-gray-00";
  const bubbleStyles = `max-w-xs md:max-w-md lg:max-w-2xl px-4 py-2 rounded-lg shadow-m ${bubbleColor}`;

  const AILogo = () => (
    <div className="w-8 h-8 rounded-full dark:bg-gray-900 flex items-center justify-center mr-3 flex-shrink-0 text-xl">
      <LogoIcon className="w-4 h-4" />
    </div>
  );

  const UserLogo = () => (
    <div className="w-8 h-8 rounded-full dark:bg-gray-900 flex items-center justify-center ml-3 flex-shrink-0">
      <UserIcon className="h-4 w-4" />
    </div>
  );

  const preprocess = (text: string) =>
  text
    .replace(/\n\n/g, '\n\n') // preserve paragraph breaks
    .replace(/^(\d+)\.\s/gm, (_, n) => `${n}. `); // ensure numbered lists are respected

  const reviewHtml = useMemo(() => {
    if (!message.content) return "";
    try {
      const raw = marked.parse(preprocess(message.content));
      return DOMPurify.sanitize(String(raw));
    } catch (e) {
      console.error("Error parsing markdown:", e);
      return DOMPurify.sanitize(String(message.content));
    }
  }, [message.content]);

  return (
    <div className={`flex items-start ${bubbleAlignment} w-full`}>
      <div className={bubbleStyles}>
        {isAI && (message.status === "pending"|| message.status === "streaming") ? (
          <LoadingSpinner />
        ) : isAI ? (
          <div className="mb-10 mt-5 flex items-start justify-start gap-1">
            {isAI && <AILogo />}
            <div
              className="prose prose-sm sm:prose-base max-w-none prose-invert
              prose-pre:bg-base-200 dark:prose-pre:bg-dark-300 prose-pre:text-base-content dark:prose-pre:text-dark-content prose-pre:p-4 prose-pre:rounded-lg prose-pre:overflow-auto prose-pre:font-mono
              prose-code:bg-base-200 dark:prose-code:bg-dark-300 prose-code:px-2 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono"
            >
              <div dangerouslySetInnerHTML={{ __html: reviewHtml }} />
            </div>
          </div>
        ) : (
          <p className="dark:text-white whitespace-pre-wrap break-words">
            {message.content}
          </p>
        )}
        {/* Optional timestamp */}
        {/* <span className="text-xs text-gray-400 mt-1 block">{new Date(message.timestamp).toLocaleTimeString()}</span> */}
      </div>
      {isUser && <UserLogo />}
    </div>
  );
};

export default MessageBubble;
