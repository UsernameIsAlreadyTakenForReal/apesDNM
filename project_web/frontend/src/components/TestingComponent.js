import { Backdrop, Button, CircularProgress, TextField } from "@mui/material";
import { Divv, RowFlex, TextFieldFlex } from "./StyledComponents";
import { useState } from "react";

export default function RoutesComponent() {
  const [loading, setLoading] = useState(false);

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

      <Divv>
        <Button onClick={() => setLoading(true)}>Click me</Button>
      </Divv>

      <div>
        <Backdrop
          sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }}
          open={loading}
          onClick={() => setLoading(false)}
        >
          <Divv>
            <CircularProgress color="inherit"> </CircularProgress>
            {/* {"Loading"} */}
          </Divv>
        </Backdrop>
      </div>
    </>
  );
}
