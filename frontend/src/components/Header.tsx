import { ThemeToggle } from "./ThemeToggle";

const Header = () => {
  return (
    <header className="static top-0 shadow-md w-full border-b border-gray-300 dark:border-gray-700/50 h-17">
      <div className="flex items-center gap-4 p-4 justify-between">
        {" "}
        <h1 className="font-medium text-xl">DAxent</h1>
        <div className="flex gap-3 items-center rounded-full">
          <button className="icon rounded-full p-1">🔍</button>
          <button title="Add document" className="icon rounded-full p-1">
            ➕
          </button>
        </div>
        <ThemeToggle />
      </div>
    </header>
  );
};

export default Header;
