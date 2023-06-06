import { useEffect } from "react";
import io from "socket.io-client";

const BASE_URL = process.env.REACT_APP_BACKEND;

const WebSocketComponent = ({ onOutputUpdated }) => {
  useEffect(() => {
    const socket = io(BASE_URL);

    socket.on("console", (data) => {
      console.log("io:", data);
      onOutputUpdated(data);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return null;
};

export default WebSocketComponent;
