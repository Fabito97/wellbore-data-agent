// src/features/chat/ChatInterface.tsx
import React, { useEffect, useRef } from "react";
import ChatInput from "./ChatInput";
import { wsService } from "../../lib/services/websocket";
import { useAppDispatch } from "../../store/hooks";
import { addMessage } from "../../store/slice/chatSlice";
import { v4 as uuidv4 } from "uuid";
import NewChat from "./NewChat";
import UserIcon from "../../components/icons/UserIcon";
import LogoIcon from "../../components/icons/LogoIcon";

export const ChatInterface: React.FC = () => {
  const dispatch = useAppDispatch();
  // // const { messages, currentSessionId, isStreaming } = useAppSelector(
  // //   (state) => state.chat
  // // );
  // const [createSession] = useCreateSessionMutation();
  // const messagesEndRef = useRef<HTMLDivElement>(null);

  // useEffect(() => {
  //   // Initialize session
  //   const initSession = async () => {
  //     const result = await createSession().unwrap();
  //     dispatch(setSessionId(result.sessionId));
  //     wsService.connect(result.sessionId);
  //   };

  //   initSession();

  //   return () => {
  //     wsService.disconnect();
  //   };
  // }, []);

  // useEffect(() => {
  //   // Auto-scroll to bottom
  //   messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  // }, [messages]);

  const handleSendMessage = (content: string) => {
    const userMessage = {
      id: uuidv4(),
      role: "user" as const,
      content,
      timestamp: Date.now(),
    };

    dispatch(addMessage(userMessage));
    wsService.sendMessage(content);
  };

  return (
    <section className="relative flex flex-col h-full">
      <div className="flex-2 p-4 sm:p-6 overflow-y-auto">
        {/* <NewChat /> */}
        <div className="space-y-6 flex flex-col items-center justify-end">
          <div className="flex items-self-start justify-self-start gap-2 text-left w-130">
            <div>
              <UserIcon className="h-6 w-6" />{" "}
            </div>
            <div className="bg-gray-700 rounded-lg p-2">
              Can you analyze this well report and tell me the expected
              production rate?
            </div>
          </div>

          {/* <div className="flex items-self-end justify-self-end gap-2 text-left w-[80%]">
            <div className="avatar ai-avatar">
              <LogoIcon className="h-6 w-6" />
            </div>
            <div className="bg-gray-700 rounded-lg p-2 ">
              <p>I've analyzed the well report. Here's what I found:</p>
              <div className="document-chip">📄 Well_Report_A123.pdf</div>

              <div className="analysis-card">
                <div className="analysis-title">📊 Extracted Parameters</div>
                <div className="param-grid">
                  <div className="param-item">
                    <div className="param-label">Measured Depth</div>
                    <div className="param-value">2,845 m</div>
                  </div>
                  <div className="param-item">
                    <div className="param-label">True Vertical Depth</div>
                    <div className="param-value">2,650 m</div>
                  </div>
                  <div className="param-item">
                    <div className="param-label">Tubing Diameter</div>
                    <div className="param-value">3.5 in</div>
                  </div>
                  <div className="param-item">
                    <div className="param-label">Reservoir Pressure</div>
                    <div className="param-value">3,200 psi</div>
                  </div>
                </div>
              </div>

              <div className="analysis-card">
                <div className="analysis-title">📈 Nodal Analysis Results</div>
                <div className="chart-placeholder">📊 IPR & VLP Curves</div>
                <p
                  style={{
                    marginTop: "12px",
                    fontSize: "14px",
                    color: "#4b5563",
                  }}
                >
                  <strong>Expected Production Rate:</strong> 1,245 bbl/day
                  <br />
                  <strong>Optimal Operating Pressure:</strong> 2,100 psi
                </p>
              </div>
            </div>
            <div className="message-time">Just now</div>
          </div> */}
        </div>
      </div>

      {/* Chat Input */}
      <ChatInput onSendMessage={handleSendMessage} isLoading={false} />
    </section>
  );
};
