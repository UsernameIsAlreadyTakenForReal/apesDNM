import { React, Component } from "react";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default class Upload extends Component {
  state = {
    file: null,
  };

  handleFile(event) {
    let file = event.target.files[0];
    this.setState({ file: file });
  }

  handleUpload() {
    let file = this.state.file;
    let fd = new FormData();

    fd.append("file", file);
    fd.append("name", "Keanu Reeves");

    fetch(BASE_URL + "upload", {
      method: "post",
      //   headers: { "content-Type": fd.type, "content-length": `${fd.size}` },
      body: fd,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
      })
      .catch((err) => {
        console.log(err);
      });
  }

  render() {
    return (
      <>
        <h1>Form</h1>
        <form onsubmit="return false">
          <label>Select file</label>
          <input
            type="file"
            name="file"
            onChange={(event) => this.handleFile(event)}
          ></input>
          <button onClick={() => this.handleUpload()}>Upload</button>
        </form>
      </>
    );
  }
}
