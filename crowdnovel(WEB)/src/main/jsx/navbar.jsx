import '../webapp/css/custom.css';
 
import React from 'react';
import { Navbar} from 'react-bootstrap';
import ReactDOM from 'react-dom';
 
class Navibar extends React.Component {
    

    render() {
        return (
          <div>
            <Navbar bg="dark" variant="dark">
              <Navbar.Brand href="#home">
                <img
                  alt=""
                  src="/logo.svg"
                  width="30"
                  height="30"
                  className="d-inline-block align-top"
                />
                {' React Bootstrap'}
              </Navbar.Brand>
            </Navbar>
          </div>
        );
    }

}
 
ReactDOM.render(<Navibar/>, document.getElementById('root'));