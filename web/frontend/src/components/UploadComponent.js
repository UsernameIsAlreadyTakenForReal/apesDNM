import { useEffect, useState } from "react";
import { Divv } from "./StyledComponents";
import {
  TextField,
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  FormHelperText,
} from "@material-ui/core";

const BASE_URL = process.env.REACT_APP_BACKEND;

async function getItems() {
  const response = await fetch(BASE_URL + "datasets");
  const data = await response.text();
  console.log(data);
}

export default function UploadComponent() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    getItems();
    console.log("hello");
  }, []);

  return (
    <>
      <Divv>
        This is the upload page. Select one of the existing datasets or choose a
        new one.
      </Divv>

      <Divv>
        <FormControl style={{ minWidth: "56%" }} variant="outlined">
          <InputLabel id="deptSelect">Datasets</InputLabel>
          <Select
            labelId="itemId"
            id="item"
            label="item select"
            onChange={() => console.log("label changed")}
          >
            <MenuItem key="0" value="" disabled>
              Choose a method
            </MenuItem>
            {items.map((item) => {
              return (
                <MenuItem key={item.id} value={item.message}>
                  {item.message}
                </MenuItem>
              );
            })}
          </Select>
          <FormHelperText>
            Please select a dataset from the ones previously processed
          </FormHelperText>
        </FormControl>
      </Divv>
    </>
  );
}
