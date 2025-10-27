import { ThemeToggle } from "../ThemeToggle";

const Header = () => {
  return (
    <header className="static top-0shadow-md w-full border-b border-gray-300 dark:border-gray-700/50">
      <div className="flex items-center gap-4 p-6 justify-between">
        {" "}
        <h1 className="font-medium text-xl">Chat with DAxent</h1>
        <div className="flex gap-3 items-center rounded-full">
          <button className=" rounded-full p-1">🔍</button>
          <button className=" rounded-full p-1">⚙️</button>
          <button className=" rounded-full p-1">👤</button>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
};

export default Header;
