css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.istockphoto.com/id/1260396908/vector/chat-bot-mascot-artificial-intelligence-virtual-assistant-innovative-technology.jpg?s=612x612&w=is&k=20&c=ycpxoRFlv-nGhLKpF5WvJZOg4dA4l0hhIFwd58szdAU=" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''


# https://thewasserstoff.com/images/wstf-logo.svg

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2018/09/15/19/23/avatar-3680134_1280.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''