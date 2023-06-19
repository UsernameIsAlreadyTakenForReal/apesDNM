import { create } from "zustand";

const useStore = create((set) => ({
  terminalFontSize: 14,
  increaseTerminalFontSize: () =>
    set((state) => {
      if (state.terminalFontSize === 24) return state;
      else return { terminalFontSize: state.terminalFontSize + 1 };
    }),
  decreaseTerminalFontSize: () =>
    set((state) => {
      if (state.terminalFontSize === 10) return state;
      else return { terminalFontSize: state.terminalFontSize - 1 };
    }),
}));

export default useStore;
