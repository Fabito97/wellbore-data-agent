import { createBrowserRouter } from "react-router-dom";
import RootLayout from "./layout/RootLayout";
import ChatPage from "./pages/ChatPage";
import NewChat from "./pages/NewChatPage";
import DocumentPage from "./pages/DocumentPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <RootLayout />,
    children: [
      { index: true, element: <NewChat /> },
      { path: "chat/:conversationId", element: <ChatPage /> },
      { path: "documents/", element: <DocumentPage /> },

      // Add more routes later
    ],
  },
]);
