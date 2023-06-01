import { Backdrop, Button, CircularProgress, TextField } from "@mui/material";
import { Divv, RowFlex, TextFieldFlex } from "./StyledComponents";
import { useState } from "react";

export default function RoutesComponent() {
  return (
    <>
      <RowFlex>
        <Divv>
          <Button
            primary
            variant="outlined"
            onClick={() => console.log("testing...")}
          >
            Click me
          </Button>
        </Divv>
      </RowFlex>
    </>
  );
}
