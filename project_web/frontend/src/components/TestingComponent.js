import { TextField } from "@mui/material";
import { Divv, RowFlex, TextFieldFlex } from "./StyledComponents";

export default function RoutesComponent() {
  return (
    <>
      <RowFlex>
        <Divv>text here</Divv>
        <TextField
          error={false}
          helperText={false ? "emptyTitleMessage" : ""}
          id="percentage-field"
          variant="outlined"
          label="Percentage of Train Data"
        />
      </RowFlex>
      <RowFlex>
        <Divv>text here</Divv>
        <TextField
          error={false}
          helperText={false ? "emptyTitleMessage" : ""}
          id="percentage-field"
          variant="outlined"
          label="Percentage of Train Data"
        />
      </RowFlex>
    </>
  );
}
