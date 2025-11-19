import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { navItems } from "../../lib/constants";
import { useChatContext } from "../../context/ChatContext";
import PlusIcon from "../ui/icons/PlusIcon";

const Sidebar = () => {
  const location = useLocation();
  const {
    startNewConversation,
    conversations,
    selectConversation,
    conversationId,
    currentConversation,
  } = useChatContext();
  const navigate = useNavigate();

 const isRouteActive = (href: string) => {
  const path = location.pathname.split('/')[1]; // first segment of path
  const hrefPath = href.split('/')[1];          // first segment of href

  // Special case: Chat nav uses href === '/' but should match both '/' and '/chat/...'
  if (href === '/') {
    return path === '' || path === 'chat';
  }

  // Default case: match first segment
  return path === hrefPath;
};

  return (
    <aside
      style={{
        background: "linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)",
      }}
      className="w-64 flex flex-col shrink-0 shadow-md border-r border-gray-700 text-gray-100"
    >
      <div className="px-4 border-b border-gray-700/50 mb-5 flex flex-col justify-center items-cente h-17">
        <div className=" flex justify-start items-center gap-2">
          <img src="favicon-32x32.png" alt="Logo" className="w-10 h-10" />
          <p className="text-xs text-cente text-gray-500 ml">
            Intelligent Analysis
          </p>
          {/* <h2></h2> */}
        </div>
      </div>

      <div className="p-5">
        <div className="flex gap-2">
          <button
            onClick={() => {
              startNewConversation();
              navigate(`/`);
            }}
            className="btn w-full rounded-lg border-gray-600 border p-1.5 px-2 bg-gray-700 flex gap-2 items-center"
          >
            <span>➕</span>
            <span className="text-xs ">New Chat</span>
          </button>

          <button
            onClick={() => {
              startNewConversation();
              navigate(`/`);
            }}
            className="btn w-full rounded-lg border-gray-600 border p-1.5 px-2 bg-gray-700 flex gap-2 items-center"
          >
            <PlusIcon />
            <span className="text-xs ">Add docs</span>
          </button>
        </div>

        {/* Navigation */}
        <div className="flex flex-col my-10">
          <h3 className="text-gray-500 text-xs font-bold mb-4">NAVIGATION</h3>
          <div className={`flex flex-col justify-startitems-center gap-2 `}>
            {navItems.map((item, idx) => (
              <NavLink
                to={item.href}
                key={idx}
                className={({ isActive }) =>
                  `${
                    idx > 1 && "hidden"
                  } py-2 px-4 rounded-lg gap-2 flex items-center text-xs ${
                    isRouteActive(item.href)
                      ? "bg-blue-700 text-white"
                      : "bg-gray-00 text-gray-300 hover:bg-blue-900"
                  }`
                }
              >
                <span>{item.icon}</span>
                <span>{item.name}</span>
              </NavLink>
            ))}
          </div>
        </div>

        {/* RECENT CHATS */}
        {conversations && conversations.length > 0 && (
          <div className="flex flex-col overflow-y-auto h-full">
            <h3 className="text-gray-500 text-xs font-bold mb-5">
              RECENT CHATS
            </h3>
            <div className="flex flex-col  gap-4 p-2 overflow-y-auto">
              {conversations.slice(0, 3)?.map((conversation) => (
                <div
                  key={conversation.id}
                  onClick={() => {
                    // selectConversation(conversation.id);
                    navigate(`/chat/${conversation.id}`, {
                      state: { conversationId: conversation.id },
                    });
                  }}
                  className={`text-xs text-gray-400 pb-2 border-b cursor-pointer ${
                    conversation.id === conversationId
                      ? "text-white border-gray-300 "
                      : "border-[#646cff] hover:border-gray-400 hover:text-gray-300/80"
                  }`}
                >
                  <button>{conversation.title}</button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
