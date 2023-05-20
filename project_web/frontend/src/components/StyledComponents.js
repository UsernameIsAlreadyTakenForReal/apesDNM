import LinearProgress from "@mui/material/LinearProgress";
import styled from "styled-components";
import Box from "@mui/material/Box";
import * as React from "react";

export const Divv = styled.div`
  margin: ${(props) => (props.margin ? props.margin : "25px")};
  font-family: ${(props) => (props.font ? props.font : "Franklin Gothic")};
  font-size: ${(props) => (props.size ? props.size : "27.5px")};
  color: ${(props) => (props.color ? props.color : "black")};

  margin-left: ${(props) =>
    props.left ? props.left : props.margin ? props.margin : "25px"};
  margin-right: ${(props) =>
    props.right ? props.right : props.margin ? props.margin : "25px"};
  margin-top: ${(props) =>
    props.top ? props.top : props.margin ? props.margin : "25px"};
  margin-bottom: ${(props) =>
    props.bottom ? props.bottom : props.margin ? props.margin : "25px"};
`;

export const ButtonDivv = styled.div`
  position: absolute;
  top: 0;
  right: 0;
  border-radius: 7px;
  border-style: solid;
  border-width: thin;
  margin: 10px;
  display: flex;
  flex-direction: column;
`;

export const MarginDivv = styled.div`
  margin-left: 10px;
  margin-right: 10px;
  margin-bottom: 10px;
  margin-top: 10px;
`;

export const TextFieldDivv = styled.div`
  margin-top: 10px;
  padding: 5px;
`;

export const Label = styled.label`
  display: inline-block;
  border: 1px solid #ccc;

  border-radius: 25px;
  padding: 12.5px 12.5px;

  padding-left: 40px;
  padding-right: 40px;

  cursor: pointer;
`;

export const TextFieldFlex = styled.div`
  border: 1px solid #ccc;
  border-radius: 25px;

  margin-left: 10px;
  margin-right: 10px;

  display: flex;
  flex-direction: row;

  justify-content: ${(props) => (props.justify ? props.justify : "center")};
  align-items: ${(props) => (props.align ? props.align : "center")};
`;

export const RowFlex = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: ${(props) => (props.justify ? props.justify : "center")};
  align-items: ${(props) => (props.align ? props.align : "center")};
`;

export const Nav = styled.nav`
  background: grey;
  height: 80px;
  display: flex;
  justify-content: space-between;
`;

export const WrapperDiv = styled.div`
  position: fixed;
  top: 42%;
  left: 42%;

  border-radius: 7px;
  border-style: solid;
  border-width: thin;

  background: white;
  // padding: 20px;

  width: 19%;
  height: 19vh;

  display: flex;
  justify-content: center;
  align-items: center;
`;

export function LinearBuffer() {
  const [progress, setProgress] = React.useState(0);
  const [buffer, setBuffer] = React.useState(10);

  const progressRef = React.useRef(() => {});
  React.useEffect(() => {
    progressRef.current = () => {
      if (progress > 100) {
        setProgress(0);
        setBuffer(10);
      } else {
        const diff = Math.random() * 10;
        const diff2 = Math.random() * 10;
        setProgress(progress + diff);
        setBuffer(progress + diff + diff2);
      }
    };
  });

  React.useEffect(() => {
    const timer = setInterval(() => {
      progressRef.current();
    }, 500);

    return () => {
      clearInterval(timer);
    };
  }, []);

  return (
    <Box sx={{ width: "100%" }}>
      <LinearProgress variant="buffer" value={progress} valueBuffer={buffer} />
    </Box>
  );
}
