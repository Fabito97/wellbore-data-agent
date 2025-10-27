const Sidebar = () => {
  return (
    <aside
      style={{
        background: "linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)",
      }}
      className="w-64 flex flex-col shrink-0 shadow-md border-r border-gray-700 text-gray-100 p-5"
    >
      <div className="pb-5 border-b border-gray-700/50 mb-5 flex flex-col gap-2">
        <div className=" flex justify-start items-center gap-2">
          <img src="favicon-32x32.png" alt="Logo" className="w-10 h-10" />
          <h2>DAxent</h2>
        </div>
        <p className="text-sm text-cente text-gray-500 ml-2">
          Intelligent Analysis
        </p>
      </div>

      <button className="w-full rounded-lg border-gray-600 border p-2 bg-gray-800 flex gap-2 items-center">
        <span>➕</span>
        <span className="text-xs ">New Chat</span>
      </button>

      {/* Navigation */}
      <div className="flex flex-col my-10">
        <h3 className="text-gray-500 text-xs font-bold mb-4">NAVIGATION</h3>
        <div className="flex flex-col justify-startitems-center gap-4">
          <a
            className="py-3 px-4 bg-blue-900 rounded-lg gap-2 flex items-center text-xs"
            href=""
          >
            <span>💬</span>
            <span>Chat</span>
          </a>
          <a
            className="py-3 px-4 bg-blue-900 rounded-lg gap-2 flex items-center text-xs"
            href=""
          >
            <span>📄</span>
            <span>Documents</span>
          </a>
          <a
            className="py-3 px-4 bg-blue-900 rounded-lg flex gap-2 items-center text-xs"
            href=""
          >
            <span>📊</span>
            <span>Analysis</span>
          </a>
        </div>
      </div>

      {/* RECENT CHATS */}
      <div className="flex flex-col">
        <h3 className="text-gray-500 text-xs font-bold mb-5">RECENT CHATS</h3>
        <div className="flex flex-col  gap-4 p-2">
          <a
            className="text-xs text-gray-400 border-b border-[#646cff] pb-2"
            href=""
          >
            Well A - Production Analysis
          </a>
          <a
            className="text-xs text-gray-400 border-b border-[#646cff] pb-2"
            href=""
          >
            NLOG Report 2024-001
          </a>
          <a
            className="text-xs text-gray-400 border-b border-[#646cff] pb-2"
            href=""
          >
            Nodal Analysis Comparison
          </a>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
