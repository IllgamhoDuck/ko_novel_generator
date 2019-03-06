import '../webapp/css/custom.css';
 
import React from 'react';
import { Button, Form, InputGroup, FormControl, Navbar, Card, Row} from 'react-bootstrap';
import ReactDOM from 'react-dom';
 
class MainPage extends React.Component {
    
    state = {
    }

    constructor(props) {
        super(props);
    
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleUserChange = this.handleUserChange.bind(this);
      }

    componentDidMount() {
        this.getTestList();
    }

    async getTestList() {
        const response = await fetch('/getContentsList');
        response.json().then((response) => this.setState({testList:response}));
    }

    handleChange(event) {
        this.setState({inputText: event.target.value});
      }
    handleUserChange(event) {
    this.setState({userName: event.target.value});
    }

    async handleSubmit(event) {
        
        let url = '/insContents';
        let options = {
                    method: 'POST',
                    mode: 'cors',
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json;charset=UTF-8'
                    },
                    body: JSON.stringify({
                        inputText: this.state.inputText,
                        userName: this.state.userName
                      })
                };
        let response = await fetch(url, options);
        let responseOK = response && response.ok;
        if (responseOK) {
            let data = await response.json();
            console.log(data)
        }
        event.preventDefault();
        
    }

    render() {
        const testList = this.state.testList;
        
        return (
            <div>
            <Navbar bg="dark" variant="dark">
              <Navbar.Brand href="#home" className="logo">
                <img
                  alt=""
                  src="img/blackoribanana_logo.png"
                  width="35"
                  height="35"
                  className="d-inline-block align-top align-center"
                />
                {' BlackOriBanana'}
              </Navbar.Brand>
            </Navbar>
            <div className="container">
                {/* {testList&&testList.map(test => { return <InputGroup>
                    <InputGroup.Prepend>
                    <InputGroup.Text>{test.USER_NAME}</InputGroup.Text>
                    </InputGroup.Prepend>
                    <FormControl as="textarea" aria-label="With textarea" value={test.TEXT} className={test.CONTENTS_TYPE} readOnly="readOnly"/>
                </InputGroup>})} */}
                <div className="cardList">
                {testList&&testList.map(test => { return <Card text="white" className={test.CONTENTS_TYPE}>
                                                            <Card.Header>{test.USER_NAME}</Card.Header>
                                                            <Card.Body>
                                                            <Card.Text>
                                                                {test.TEXT}
                                                            </Card.Text>
                                                            </Card.Body>
                                                        </Card>})}
                </div>
                <Form onSubmit={this.handleSubmit}>
                    <Form.Group controlId="form">
                        <Form.Control placeholder="사용자명" as="input" onChange={this.handleUserChange}></Form.Control>
                        <Form.Control placeholder="내용" as="textarea" rows="3" onChange={this.handleChange}/>
                        <Button variant="primary" type="submit">
                            보내기
                        </Button>
                    </Form.Group>
                </Form>
            </div>
            </div>
        );
    }

}
 
ReactDOM.render(<MainPage/>, document.getElementById('root'));