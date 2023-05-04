import styled from "styled-components";

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

export const RowFlex = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: ${(props) => (props.justify ? props.justify : "center")};
  align-items: ${(props) => (props.align ? props.align : "center")};
`;

export const TextFieldDivv = styled.div`
  margin-top: 10px;
  padding: 5px;
`;
