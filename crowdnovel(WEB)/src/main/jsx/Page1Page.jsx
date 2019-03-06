import '../webapp/css/custom.css';
 
import React from 'react';
import ReactDOM from 'react-dom';
 
class Page1Page extends React.Component {
 
    render() {
        return <div className="page1">Page1 페이지</div>;
    }
 
}
 
ReactDOM.render(<Page1Page/>, document.getElementById('root'));