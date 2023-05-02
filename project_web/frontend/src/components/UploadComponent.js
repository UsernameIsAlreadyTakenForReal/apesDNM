import { useEffect, useState } from "react";
import { Divv, RowFlex } from "./StyledComponents";
import {
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from "@mui/material";

import { Input } from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [items, setItems] = useState([]);
  const [singularItem, setSingularItem] = useState([]);

  const [hover1, setHover1] = useState(false);
  const [value, setValue] = useState("");

  async function getItems() {
    const response = await fetch(BASE_URL + "datatypes", {
      method: "get",
    });
    const data = await response.json();
    setItems(data);
  }

  async function getSingularItem() {
    if (value === "") return;
    console.log(BASE_URL + "datatype?id=" + value);
    const response = await fetch(BASE_URL + "datatype?id=" + value, {
      method: "post",
    });
    const data = await response.text();
    setSingularItem(data);
  }

  useEffect(() => {
    getItems();
  }, []);

  return (
    <>
      <Divv>Select one of the existing types of data.</Divv>

      <RowFlex justify="left">
        <FormControl sx={{ width: "30%", margin: "20px" }} variant="outlined">
          <InputLabel id="itemSelect">Data type</InputLabel>
          <Select
            labelId="itemId"
            id="item"
            label="itemSelect"
            onChange={(event) => {
              console.log("Now selected", event.target.value);
              setValue(event.target.value);
            }}
          >
            <MenuItem key="0" value="" disabled>
              Choose a method
            </MenuItem>
            {items.map((item) => {
              return (
                <MenuItem key={item.id} value={item.id}>
                  {item.method}
                </MenuItem>
              );
            })}
          </Select>
          {/* <FormHelperText>
              Please select a data type from the ones previously processed
            </FormHelperText> */}
        </FormControl>
      </RowFlex>

      <input
        accept="image/*"
        className={classes.input}
        style={{ display: "none" }}
        id="raised-button-file"
        multiple
        type="file"
      />
      <label htmlFor="raised-button-file">
        <Button variant="raised" component="span" className={classes.button}>
          Upload
        </Button>
      </label>
      <Divv>
        <Button
          style={{
            background: hover1 === false ? "black" : "orange",
            color: hover1 === false ? "white" : "black",
            fontWeight: "bold",
          }}
          variant="contained"
          color="primary"
          size="large"
          onClick={() => {
            if (value === "") return;
            console.log("Fetching request for method", value);

            getSingularItem();
          }}
          onMouseEnter={() => {
            setHover1(true);
          }}
          onMouseLeave={() => {
            setHover1(false);
          }}
        >
          Upload data
        </Button>
      </Divv>

      {/* {singularItem.length ? JSON.stringify(singularItem) : <></>} */}
    </>
  );
}
